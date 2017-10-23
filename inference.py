from __future__ import print_function

import argparse
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np

from model import ICNet
from tools import decode_labels

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
input_size = [1024, 2048]
num_classes = 19

model_train30k = './model/icnet_cityscapes_train_30k.npy'
model_trainval90k = './model/icnet_cityscapes_trainval_90k.npy'
model = 'train'

SAVE_DIR = './output/'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--train-model", action="store_true",
                        help="restore from train_30k.npy")
    parser.add_argument("--trainval-model", action="store_true",
                        help="restore from trainval_90k.npy")  
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")

    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    ext = filename.split('.')[-1]

    if ext.lower() == 'png':
        img = tf.image.decode_png(tf.read_file(img_path), channels=3)
    elif ext.lower() == 'jpg':
        img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    else:
        print('cannot process {0} file.'.format(file_type))

    return img, filename

def preprocess(img):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    
    img.set_shape([1024, 2048, 3]) 
    img = tf.expand_dims(img, dim=0)

    return img

def main():
    args = get_arguments()
    
    img, filename = load_img(args.img_path)
    img = preprocess(img)
    
    # Create network.
    net = ICNet({'data': img}, num_classes=num_classes)
    raw_output = net.layers['conv6_cls']
    
    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=input_size, align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()
    
    if args.train_model:
        print('Restore from train30k model...')
        net.load(model_train30k, sess)
    elif args.trainval_model:
        print('Restore from trainval90k model...')
        net.load(model_trainval90k, sess)
    else:
        print('You didnot specify model to restore, restore from trainval90k model by default...')
        net.load(model_trainval90k, sess)

    preds = sess.run(pred)

    msk = decode_labels(preds, num_classes=num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    im.save(args.save_dir + filename)

if __name__ == '__main__':
    main()
