# ICNet_tensorflow
## Introduction
  This is an implementation of ICNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/ICNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.

## Install
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyadTdHejU1Q1lfRkU?usp=sharing
) and put into `model` directory.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png
```
List of Args:
```
--model=train    - To select train_30k model (Default)
--model=trainval - To select trainval_90k model
```
Inference time:  ~0.02s, I have no idea why it's faster than caffe implementation 

## Evaluation
Perform in single-scaled model on the cityscapes validation datase.

| Model | Accuracy |  Missing accuracy |
|:-----------:|:----------:|:---------:|
| train_30k   | **65.3/67.7** | **2.4%** |
| trainval_90k| **78.06%**    | None |

To get evaluation result, you need to download Cityscape dataset from [Official website](https://www.cityscapes-dataset.com/) first. Then change `DATA_DIRECTORY` to your dataset path in `evaluate.py`:
```
DATA_DIRECTORY = /Path/to/dataset
```

Then run the following command: 
```
python evaluate.py
```
List of Args:
```
--model=train    - To select train_30k model (Default)
--model=trainval - To select trainval_90k model
--measure-time   - Calculate inference time (e.q subtract preprocessing time)
```
## Image Result
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/test.png)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/test.png)
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/test2.png)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/test2.png)

## Citation
    @article{zhao2017icnet,
      author = {Hengshuang Zhao and
                Xiaojuan Qi and
                Xiaoyong Shen and
                Jianping Shi and
                Jiaya Jia},
      title = {ICNet for Real-Time Semantic Segmentation on High-Resolution Images},
      journal={arXiv preprint arXiv:1704.08545},
      year = {2017}
    }
