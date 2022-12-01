import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm
import deepspeed
import os

__all__ = ["DEFAULT_TIMEOUT", "launch_deepspeed", "launch_deepspeed_multinodes"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch_deepspeed(
        main_func,
        num_gpus_per_machine,
        num_machines=1,
        machine_rank=0,
        dist_url=None,
        args=(),
        timeout=DEFAULT_TIMEOUT,
):
    """
    Modified by Jialian Wu from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:

        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
        local_rank,
        main_func,
        world_size,
        num_gpus_per_machine,
        machine_rank,
        dist_url,
        args,
        timeout=DEFAULT_TIMEOUT,
):
    '''
    Modified by Jialian Wu from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py
    Adaptation for deepspeed
    '''
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    assert dist_url.startswith('tcp://')
    master_address = dist_url.split('tcp://')[1].split(':')[0]
    master_port = dist_url.split('tcp://')[1].split(':')[1]

    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = master_port
    try:
        deepspeed.init_distributed()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*args)


def get_mpi_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))


def get_mpi_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))


def get_mpi_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))


def launch_deepspeed_multinodes(
        main_func,
        dist_url=None,
        args=(),
):
    """
        Launch multi-node training via deepspeed.
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    assert dist_url.startswith('tcp://')
    master_address = dist_url.split('tcp://')[1].split(':')[0]
    master_port = dist_url.split('tcp://')[1].split(':')[1]

    os.environ['RANK'] = str(get_mpi_rank())
    os.environ['LOCAL_RANK'] = str(get_mpi_local_rank())
    os.environ['WORLD_SIZE'] = str(get_mpi_size())
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = master_port
    try:
        deepspeed.init_distributed()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    torch.cuda.set_device(get_mpi_local_rank())

    comm.synchronize()

    main_func(*args)
