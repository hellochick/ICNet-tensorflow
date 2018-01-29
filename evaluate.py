from __future__ import print_function
import argparse
import os
import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from model import ICNet, ICNet_BN
from image_reader import read_labeled_image_list

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# define setting & model configuration
ADE20k_param = {'name': 'ade20k',
                'input_size': [480, 480],
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'num_steps': 2000,
                'data_dir': '../../ADEChallengeData2016/', 
                'data_list': './list/ade20k_val_list.txt'}
                
cityscapes_param = {'name': 'cityscapes',
                    'input_size': [1025, 2049],
                    'num_classes': 19,
                    'ignore_label': 255,
                    'num_steps': 500,
                    'data_dir': '/data/cityscapes_dataset/cityscape', 
                    'data_list': './list/cityscapes_val_list.txt'}

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy', 
              'trainval': './model/icnet_cityscapes_trainval_90k.npy',
              'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
              'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
              'others': './model/'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--measure-time", action="store_true",
                        help="whether to measure inference time")
    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=True)
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

time_list = []
def calculate_time(sess, net, pred, feed_dict):
    start = time.time()
    sess.run(net.layers['data'], feed_dict=feed_dict)
    data_time = time.time() - start

    start = time.time()
    sess.run(pred, feed_dict=feed_dict)
    total_time = time.time() - start

    inference_time = total_time - data_time

    time_list.append(inference_time)
    print('average inference time: {}'.format(np.mean(time_list)))

def preprocess(img, param):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    shape = param['input_size']

    if param['name'] == 'cityscapes':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
    elif param['name'] == 'ade20k':
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_bilinear(img, shape, align_corners=True)
        
    return img

def main():
    args = get_arguments()
    
    if args.dataset == 'ade20k':
        param = ADE20k_param
    elif args.dataset == 'cityscapes':
        param = cityscapes_param

    # Set placeholder
    image_filename = tf.placeholder(dtype=tf.string)
    anno_filename = tf.placeholder(dtype=tf.string)

    # Read & Decode image
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    ori_shape = tf.shape(img)
    img = preprocess(img, param)

    model = model_config[args.model]
    net = model({'data': img}, num_classes=param['num_classes'], 
                    filter_scale=args.filter_scale, evaluation=True)

    # Predictions.
    raw_output = net.layers['conv6_cls']

    raw_output_up = tf.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    raw_pred = tf.expand_dims(raw_output_up, dim=3)

    # mIoU
    pred_flatten = tf.reshape(raw_pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])

    mask = tf.not_equal(raw_gt, param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if args.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes']+1)
    elif args.dataset == 'cityscapes':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    
    sess.run(init)
    sess.run(local_init)

    model_path = model_paths[args.model]
    if args.model == 'others':
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=tf.global_variables())
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        net.load(model_path, sess)
        print('Restore from {}'.format(model_path))

    img_files, anno_files = read_labeled_image_list(param['data_dir'], param['data_list'])
    for i in trange(param['num_steps'], desc='evaluation', leave=True):
        feed_dict = {image_filename: img_files[i], anno_filename: anno_files[i]}
        _ = sess.run(update_op, feed_dict=feed_dict)

        if i > 0 and args.measure_time:
            calculate_time(sess, net, raw_pred, feed_dict)

    print('mIoU: {}'.format(sess.run(mIoU)))
   

if __name__ == '__main__':
    main()
