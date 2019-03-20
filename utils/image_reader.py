import os
import numpy as np
import tensorflow as tf
import glob
import cv2

def read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")

        image = os.path.join(data_dir, image)
        mask = os.path.join(data_dir, mask)
        mask = mask.strip()
        
        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image)

        if not tf.gfile.Exists(mask):
            raise ValueError('Failed to find file: ' + mask)

        images.append(image)
        masks.append(mask)

    return images, masks

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, axis=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch

def _extract_mean(img, img_mean, swap_channel=False):
    # swap channel and extract mean
    
    if swap_channel:
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)

    img -= img_mean
    
    return img

def _parse_function(image_filename, label_filename, img_mean):
    img_contents = tf.read_file(image_filename)
    label_contents = tf.read_file(label_filename)
       
    # Decode image & label
    img = tf.image.decode_jpeg(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)
    
    # swap channel and extract mean
    img = _extract_mean(img, img_mean, swap_channel=True)

    return img, label

def _image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)

    return img, label

def _image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=[0])

    return img, label

def _random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label):
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(
                            combined,
                            0,
                            0,
                            tf.maximum(crop_h, image_shape[0]),
                            tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))

    return img_crop, label_crop

def _check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]
    
    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h/32) + 1) * 32
        new_w = (int(ori_w/32) + 1) * 32
        shape = [new_h, new_w]
        
        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
        
        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape

def _infer_preprocess(img, swap_channel=False):
    o_shape = img.shape[0:2]
        
    img = _extract_mean(img, swap_channel)
    img = tf.expand_dims(img, axis=0)
    img, n_shape = _check_input(img)
        
    return img, o_shape, n_shape

def _eval_preprocess(img, label, shape, dataset):
    if dataset == 'cityscapes':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
    else:
        img = tf.image.resize_images(img, shape, align_corners=True)
     
    return img, label

class ImageReader(object):
    '''
    Generic ImageReader which reads images and corresponding segmentation masks
    from the disk, and enqueues them into a TensorFlow queue using tf.Dataset API.
    '''

    def __init__(self, cfg, img_path=None, mode='eval'):
        if mode == 'train' or mode == 'eval':
            self.image_list, self.label_list = read_labeled_image_list(cfg.param['data_dir'], cfg.param[mode+'_list'])
            print(cfg.param[mode+'_list'])
            self.dataset = self.create_tf_dataset(cfg)

            self.next_image, self.next_label = self.dataset.make_one_shot_iterator().get_next() 
    
    def create_tf_dataset(self, cfg):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_list, self.label_list))
        dataset = dataset.map(lambda x, y: _parse_function(x, y, cfg.IMG_MEAN), num_parallel_calls=cfg.N_WORKERS)

        if cfg.is_training: # Training phase
            h, w = cfg.TRAINING_SIZE

            if cfg.random_scale:
                dataset = dataset.map(_image_scaling, num_parallel_calls=cfg.N_WORKERS)
            if cfg.random_mirror:
                dataset = dataset.map(_image_mirroring, num_parallel_calls=cfg.N_WORKERS)

            dataset = dataset.map(lambda x, y: 
                                  _random_crop_and_pad_image_and_labels(x, y, h, w, cfg.param['ignore_label']),
                                  num_parallel_calls=cfg.N_WORKERS)
            
            dataset = dataset.shuffle(buffer_size=500)
            dataset = dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
            dataset = dataset.repeat()
            
        else: # Evaluation phase            
            dataset = dataset.map(lambda x, y: 
                                  _eval_preprocess(x, y, cfg.param['eval_size'], cfg.dataset),
                                  num_parallel_calls=cfg.N_WORKERS)
            dataset = dataset.batch(1)
                
        return dataset

    
    
