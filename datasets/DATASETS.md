# Dataset preparation


## COCO Dataset

- Download the coco 2017 dataset from the [official website](https://cocodataset.org/#download).

Dataset strcture should look like:
  ~~~
  ${GRiT_ROOT}
  |-- datasets
  `-- |-- coco
      |-- |-- train2017/
      |-- |-- val2017/
      |-- |-- test2017/
      |-- |-- annotations/
      |-- |-- |-- instances_train2017.json
      |-- |-- |-- instances_val2017.json
      |-- |-- |-- image_info_test-dev2017.json
  ~~~

## VG Dataset
- Download images from [official website](https://visualgenome.org/api/v0/api_home.html)
- Download our pre-processed annotations: 
  [train.json](https://datarelease.blob.core.windows.net/grit/VG_preprocessed_annotations/train.json) and
  [test.json](https://datarelease.blob.core.windows.net/grit/VG_preprocessed_annotations/test.json)

Dataset strcture should look like:
  ~~~
  ${GRiT_ROOT}
  |-- datasets
  `-- |-- vg
      |-- |-- images/
      |-- |-- annotations/
      |-- |-- |-- train.json
      |-- |-- |-- test.json
  ~~~

## References
Please cite the corresponding references if you use the datasets.

~~~
 @inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}

 @article{krishna2017visual,
  title={Visual genome: Connecting language and vision using crowdsourced dense image annotations},
  author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and others},
  journal={International journal of computer vision},
  volume={123},
  number={1},
  pages={32--73},
  year={2017},
  publisher={Springer}
}
~~~