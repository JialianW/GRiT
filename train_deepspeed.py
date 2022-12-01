# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jialian Wu from https://github.com/facebookresearch/Detic/blob/main/train_net.py
import logging
import os
import sys
from collections import OrderedDict
import torch
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config

from grit.config import add_grit_config
from grit.data.custom_build_augmentation import build_custom_augmentation
from grit.data.custom_dataset_dataloader import  build_custom_train_loader
from grit.data.custom_dataset_mapper import CustomDatasetMapper
from grit.custom_solver import build_custom_optimizer

from grit.evaluation.eval import GRiTCOCOEvaluator, GRiTVGEvaluator
import deepspeed
from lauch_deepspeed import launch_deepspeed, launch_deepspeed_multinodes


logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
            cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == 'coco':
            evaluator = GRiTCOCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'vg':
            evaluator = GRiTVGEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise NotImplementedError('We have not implemented the evaluator for {}'.format(evaluator_type))

        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def get_deep_speed_config(train_batch_size):
    config_params = {
        'train_batch_size': train_batch_size,
        'steps_per_print': 100000000,
        'logging': {'steps_per_print': 200000,},
        'zero_optimization': {'stage': 1,},
        'zero_allow_untested_optimizer': True,
        'fp16': {'enabled': True,},
        'flops_profiler': {
            'enabled': False,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 1,
            'detailed': True
        }
    }

    return config_params


def get_master_node_ip():

    if 'AZ_BATCH_HOST_LIST' in os.environ:
        ret = os.environ['AZ_BATCH_HOST_LIST'].split(',')[0]
    elif 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
        ret = os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
    elif 'MASTER_IP' in os.environ:
        ret = os.environ['MASTER_IP']
    else:
        raise NotImplementedError
    import socket
    return socket.gethostbyname(ret)


def do_train(cfg, model, resume=False, train_batch_size=1):
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = checkpointer.resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    model, optimizer, _, scheduler = deepspeed.initialize(
        config_params=get_deep_speed_config(train_batch_size),
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    model.train()

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = CustomDatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    data_loader = build_custom_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            model.backward(losses)
            model.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()

            if (cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                    (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.output_dir_name:
        cfg.OUTPUT_DIR = args.output_dir_name
    logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), color=False, name="grit")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)

    if args.num_machines == 1:
        train_batch_size = args.num_gpus_per_machine * cfg.DATALOADER.DATASET_BS
    else:
        train_batch_size = args.num_machines * args.num_gpus_per_machine * cfg.DATALOADER.DATASET_BS
    logger.info('Training batch size: {}'.format(train_batch_size))
    do_train(cfg, model, resume=args.resume, train_batch_size=train_batch_size)
    return


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--output-dir-name", type=str, default='./output/GRiT')
    args.add_argument("--num-gpus-per-machine", type=int, default=8)
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(12345)
        print("Command Line Args:", args)
        launch_deepspeed(
            main,
            args.num_gpus_per_machine,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
    else:
        print('Multi-nodes Training')
        args.dist_url = 'tcp://{}:{}'.format(get_master_node_ip(), 12345)
        print("Command Line Args:", args)
        launch_deepspeed_multinodes(
            main,
            dist_url=args.dist_url,
            args=(args,),
        )
