import os

import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf

def process_image(image_file):
    """Decode image at given path."""
    # Method 1: return <class 'tf.Tensor'>
    image_string = tf.io.read_file(image_file)

    # Method 2: return <class 'bytes'>
    # with open(image_file, 'rb') as f:
    #     image_string = f.read() # binary-string

    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        return 0, image_string
    except tf.errors.InvalidArgumentError:
        print('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string


def parse_annot(annot_file, classes_map):
    """Parse VOC format annotations."""
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
    classes_name, classes_id = [], []

    for obj in root.iter('object'):
        label = obj.find('name').text

        if len(classes_map) > 0 and label not in classes_map:
            continue
        else:
            classes_id.append(classes_map[label])
            classes_name.append(label.encode('utf8'))

        for box in obj.findall('bndbox'):
            xmin.append(float(box.find('xmin').text) / width)
            ymin.append(float(box.find('ymin').text) / height)
            xmax.append(float(box.find('xmax').text) / width)
            ymax.append(float(box.find('ymax').text) / height)

    image_info['filename'] = file_name
    image_info['width'] = width
    image_info['height'] = height
    image_info['depth'] = depth
    image_info['classes_id'] = classes_id
    image_info['classes_name'] = classes_name
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax

    image_info_list.append(image_info)

    return image_info_list


def create_tf_example(image_string, image_info_list):
    """Convert VOC ground truth to TFExample protobuf."""
    for info in image_info_list:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes_id = info['classes_id']
        classes_name = info['classes_name']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']

    if isinstance(image_string, type(tf.constant(0))):
        encoded_image = [image_string.numpy()]
    else:
        encoded_image = [image_string]

    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    feature_dict = {
        'filename':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes_id':
            tf.train.Feature(int64_list=tf.train.Int64List(value=classes_id)),
        'classes_name':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_name)),
        'x_mins':
            tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'y_mins':
            tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'x_maxes':
            tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'y_maxes':
            tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image_raw':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example # example.SerializeToString()
