import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import read_classes

ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data")
DATA_SET_PATH = os.path.join(DATA_PATH, "VOCdevkit/VOC2012")
ANNOTATIONS_PATH = os.path.join(DATA_SET_PATH, "Annotations")
IMAGES_PATH = os.path.join(DATA_SET_PATH, "JPEGImages")

# Classes that you want to detect.
CLASSES = read_classes("./model_data/voc2012_classes.txt")

def process_image(image_file):
    """Decode image at given path."""
    # Method 1: return <class 'tf.Tensor'>
    image_string = tf.io.read_file(image_file)

    # Method 2: return <class 'bytes'>
    #with open(image_file, 'rb') as f:
    #    image_string = f.read() # binary-string

    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        #image_data = tf.image.resize(image_data, [300, 300])
        #image_data /= 255.0 # normalize to [0, 1] range
        return 0, image_string, image_data
    except tf.errors.InvalidArgumentError:
        print('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string, None

def parse_annot(annot_file):
    """Parse Pascal VOC annotations."""
    tree = ET.parse(annot_file)
    root = tree.getroot()

    image_info = {}
    image_info_list = []

    file_name = root.find('filename').text

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    xmin, ymin, xmax, ymax = [], [], [], []
    classes = []

    for obj in root.iter('object'):
        label = obj.find('name').text

        if len(CLASSES) > 0 and label not in CLASSES:
            continue
        else:
            classes.append(CLASSES.index(label))

        for box in obj.findall('bndbox'):
            xmin.append(float(box.find('xmin').text) / width)
            ymin.append(float(box.find('ymin').text) / height)
            xmax.append(float(box.find('xmax').text) / width)
            ymax.append(float(box.find('ymax').text) / height)

    image_info['filename'] = file_name
    image_info['width'] = width
    image_info['height'] = height
    image_info['depth'] = depth
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax

    image_info_list.append(image_info)

    return image_info_list

def convert_voc_to_tf_example(image_string, image_info_list):
    """Convert Pascal VOC ground truth to TFExample protobuf."""
    for info in image_info_list:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes = info['class']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']

    if isinstance(image_string, type(tf.constant(0))):
        encoded_image = [image_string.numpy()]
    else:
        encoded_image = [image_string]

    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes':tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'x_mins':tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'y_mins':tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'x_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'y_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))

    return example # example.SerializeToString()

def main():
    images = sorted(glob.glob(os.path.join(IMAGES_PATH, '*.jpg')))
    annots = sorted(glob.glob(os.path.join(ANNOTATIONS_PATH, '*.xml')))
    train_file = 'data/train_voc.tfrecord'
    counter = 0
    skipped = 0

    with tf.io.TFRecordWriter(train_file) as writer:
        for image, annot in (zip(images, annots)):
            # processes the image and parse the annotation
            error, image_string, image_data = process_image(image)
            image_info_list = parse_annot(annot)

            if not error:
                # convert voc to `tf.Example`
                example = convert_voc_to_tf_example(image_string, image_info_list)

                # write the `tf.example` message to the TFRecord files
                writer.write(example.SerializeToString())
                counter += 1
                print('{} : Processed {:d} of {:d} images.'.format(
                    datetime.now(), counter, len(images)))
            else:
                skipped += 1
                print('{} : Skipped {:d} of {:d} images.'.format(
                    datetime.now(), skipped, len(images)))

    print('{} : Wrote {} images to {}'.format(
            datetime.now(), counter, train_file))

if __name__ == "__main__":
    main()
