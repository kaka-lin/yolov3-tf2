import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from models.yolov3 import (
    Yolov3,
    yolo_eval
)
from utils.yolo_utils import (
    read_classes,
    read_anchors,
    generate_colors,
    draw_outputs
)
from utils.common import preprocess_image

flags.DEFINE_integer('size', 416, 'the input size for model')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './model_data/coco_classes.txt',
                    'path to classes file')
flags.DEFINE_string('anchors', './model_data/yolov3_anchors.txt',
                    'path to anchors file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_float('iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('score_threshold', 0.5, 'score threshold')
flags.DEFINE_boolean('keep_aspect_ratio', True, 'resize image with unchanges/changed aspect ratio')
flags.DEFINE_string('image', './data/street.jpg', 'path to input image')
flags.DEFINE_string('output', './out/output.png', 'path to output image')
flags.DEFINE_boolean('save', False, 'save image or not')


def main(argv):
    yolo = Yolov3(classes=FLAGS.num_classes)
    try:
        yolo.load_weights(FLAGS.weights).expect_partial()
        logging.info('Weights loaded')
    except ValueError as err:
        logging.error(
            '[Error] Number of classes is not the same. {}'.format(err))
        sys.exit()

    # Load classes and anchors
    class_names = read_classes(FLAGS.classes)
    # NOTE: no normalize
    anchors = read_anchors(FLAGS.anchors)
    logging.info('Classes and Anchors loaded')

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)

    # NOTE: 1. Open image with `OpenCV` or `tf.image.decode_image()`
    #          would cause different result.
    #       2. And need to notice related color space (RGB, BGR, YUV, etc.)

    # 1. 使用 OpenCV 讀入圖像 (BGR)
    # open_type = "cv"
    # image = cv2.imread(FLAGS.image) # 載入圖像

    # 2. 使用 Tensorflow 讀入圖像 (RGB)
    open_type = "tf"
    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)
    image = img_raw.numpy()

    # 進行圖像輸入的前處理
    image_shape = image.shape[:2] # h, w
    input_dims = (FLAGS.size, FLAGS.size)
    input_image = preprocess_image(image, input_dims,
                                   open_type=open_type,
                                   keep_aspect_ratio=FLAGS.keep_aspect_ratio)

    # 進行圖像偵測
    yolo_outputs = yolo.predict(input_image)
    scores, boxes, classes = yolo_eval(
        yolo_outputs,
        anchors,
        image_shape=image_shape,
        input_dims=input_dims,
        letterbox=FLAGS.keep_aspect_ratio,
        classes=FLAGS.num_classes,
        score_threshold=FLAGS.score_threshold,
        iou_threshold=FLAGS.iou_threshold
    )

    logging.info("detections:")
    for i in range(scores.shape[0]):
        logging.info("\t{}, {}, {}".format(
            class_names[int(classes[i])], scores[i], boxes[i]
        ))

    # Draw bounding boxes on the image file
    image = draw_outputs(image, (scores, boxes, classes), class_names, colors)

    # Show
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

    # Save
    if FLAGS.save:
        out_path = '/'.join(FLAGS.output.split('/')[:-1])
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.output, image)


if __name__ == "__main__":
    app.run(main)
