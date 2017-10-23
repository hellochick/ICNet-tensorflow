# ICNet_tensorflow
## Introduction
  This is an implementation of ICNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/ICNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.

## Install
1. Get restore checkpoint from [Google Drive](https://drive.google.com/file/d/0B6VgjAr4t_oTTDh2SVJIa2VkZVU/view?usp=sharing) and put into `model` directory.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png
```
Inference time:  ~0.02s, I have no idea why it's faster than caffe implementation 

## Evaluation
Perform in single-scaled model on the cityscapes validation datase.

| Model | Accuracy |  Missing accuracy |
|:-----------:|:----------:|:---------:|
| train_30k   | **65.3/67.7** | **2.4% |
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
--train-model    - To select train_30k model
--trainval-model - To select trainval_90k model
--measure-time   - Calculate inference time (e.q subtract preprocessing time)
```
## Image Result
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet/blob/master/input/test.png)  |  ![](https://github.com/hellochick/ICNet/blob/master/output/test.png)
![](https://github.com/hellochick/ICNet/blob/master/input/test2.png)  |  ![](https://github.com/hellochick/ICNet/blob/master/output/test2.png)
