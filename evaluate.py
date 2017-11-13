from __future__ import print_function
import argparse
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np

from model import ICNet, ICNet_BN
from tools import decode_labels
from image_reader import ImageReader

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
input_size = [1024, 2048]

SAVE_DIR = './output/'

DATA_DIRECTORY = '/data/cityscapes_dataset/cityscape'
DATA_LIST_PATH = './list/eval_list.txt'

model_train30k = './model/icnet_cityscapes_train_30k.npy'
model_trainval90k = './model/icnet_cityscapes_trainval_90k.npy'
model_train30k_bn = './model/icnet_cityscapes_train_30k_bnnomerge.npy'
model_trainval90k_bn = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'

num_classes = 19
ignore_label = 255 # Don't care label
num_steps = 500 # numbers of image in validation set
time_list = []

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--measure-time", action="store_true",
                        help="whether to measure inference time")
    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn'],
                        required=True)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def calculate_time(sess, net):
    start = time.time()
    sess.run(net.layers['data'])
    data_time = time.time() - start

    start = time.time()
    sess.run(net.layers['conv6_cls'])
    total_time = time.time() - start

    inference_time = total_time - data_time

    time_list.append(inference_time)
    print('average inference time: {}'.format(np.mean(time_list)))

def main():
    args = get_arguments()
    print(args)

    coord = tf.train.Coordinator()

    tf.reset_default_graph()
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_DIRECTORY,
            DATA_LIST_PATH,
            input_size,
            ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    if args.model[-2:] == 'bn':
        net = ICNet_BN({'data': image_batch}, num_classes=num_classes)
    else:
        net = ICNet({'data': image_batch}, num_classes=num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['conv6_cls']

    raw_output_up = tf.image.resize_bilinear(raw_output, size=input_size, align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(label_batch, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    restore_var = tf.global_variables()

    if args.model == 'train':
        print('Restore from train30k model...')
        net.load(model_train30k, sess)
    elif args.model == 'trainval':
        print('Restore from trainval90k model...')
        net.load(model_trainval90k, sess)
    elif args.model == 'train_bn':
        print('Restore from train30k bnnomerge model...')
        net.load(model_train30k_bn, sess)
    elif args.model == 'trainval_bn':
        print('Restore from trainval90k bnnomerge model...')
        net.load(model_trainval90k_bn, sess)
        
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(num_steps):
        preds, _ = sess.run([pred, update_op])

        if step > 0 and args.measure_time:
            calculate_time(sess, net)

        if step % 10 == 0:
            print('Finish {0}/{1}'.format(step, num_steps))

    print('step {0} mIoU: {1}'.format(step, sess.run(mIoU)))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
