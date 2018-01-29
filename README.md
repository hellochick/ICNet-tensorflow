# ICNet_tensorflow
## Introduction
  This is an implementation of ICNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/ICNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.
  
## Update
#### 2018/1/27:
1. Improve evaluation results by changing `interp` operation and add `zero padding` in front of max pooling layer. Such modification improve the mIoU to **67.35%** ( much closer to original work ).  [Pull request #35](https://github.com/hellochick/ICNet-tensorflow/pull/35)

#### 2017/11/15:
1. Support `training phase`, you can train on your own dataset. Please read the guide below.

#### 2017/11/13:
1. Add `bnnomerge model` which reparing for training phase. Choose different model using flag `--model=train, train_bn, trainval, trainval_bn` (Upload model in google drive).
2. Change `tf.nn.batch_normalization` to `tf.layers.batch_normalization`.

#### 2017/11/07:
`Support every image size larger than 128x256` by changing the avg pooling ksize and strides in the pyramid module. If input image size cannot divided by 32, it will be padded in to mutiple of 32.


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
--model=train       - To select train_30k model (Default)
--model=trainval    - To select trainval_90k model
--model=train_bn    - To select train_30k_bn model
--model=trainval_bn - To select trainval_90k_bn model
--model=others      - To select your own checkpoint
```
### Inference time
* **Including time of loading images**: ~0.04s
* **Excluding time of loading images (Same as described in paper)**: ~0.03s

## Evaluation
Perform in single-scaled model on the cityscapes validation datase.

| Model | Accuracy |  Missing accuracy |
|:-----------:|:----------:|:---------:|
| train_30k   | **67.35/67.7** | **0.35%** |
| trainval_90k| **81.06%**    | None |

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
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/test_1024x2048.png)  |  ![](https://github.com/hellochick/ICNet_tensorflow/blob/master/output/test_1024x2048.png)

## Training on your own dataset
> Note: This implementation is different from the details descibed in ICNet paper, since I did not re-produce model compression part. Instead, we train on the half kernel directly.

### Step by Step
**1. Change the `DATA_LIST_PATH`** in line 22, make sure the list contains the absolute path of your data files, in `list.txt`:
```
/ABSOLUTE/PATH/TO/image /ABSOLUTE/PATH/TO/label
```
**2. Set Hyperparameters** (line 21-35) in `train.py`
```
BATCH_SIZE = 48
IGNORE_LABEL = 0
INPUT_SIZE = '480,480'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 27
NUM_STEPS = 60001
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
```
Also **set the loss function weight** (line 38-40) descibed in the paper:
```
# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0
```
**3.** Run following command and **decide whether to update mean/var or train beta/gamma variable**.
```
python train.py --update-mean-var --train-beta-gamma
```
After training the dataset, you can run following command to get the result:  
```
python inference.py --img-path=YOUR_OWN_IMAGE --model=others
```
### Result ( inference with my own data )

Input                      |  Output
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/indoor1.jpg)  |  ![](https://github.com/hellochick/ICNet-tensorflow/blob/master/output/indoor1.jpg)
![](https://github.com/hellochick/ICNet_tensorflow/blob/master/input/indoor3.jpg)  |  ![](https://github.com/hellochick/ICNet-tensorflow/blob/master/output/indoor3.jpg)


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
