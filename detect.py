import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3.model import Yolov3, yolo_eval
from yolov3.utils import *

flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './model_data/coco_classes.txt',
                    'path to classes file')
flags.DEFINE_string('anchors', './model_data/yolov3_anchors.txt',
                    'path to anchors file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_float('iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('score_threshold', 0.5, 'score threshold')
flags.DEFINE_string('image', './data/dog.jpg', 'path to input image')
flags.DEFINE_string('output', './out/output.jpg', 'path to output image')
flags.DEFINE_boolean('save', False, 'save image or not')

def main(argv):
    yolo = Yolov3(classes=FLAGS.num_classes)
    try:
        yolo.load_weights(FLAGS.weights).expect_partial()
        logging.info('Weights loaded')
    except ValueError as err:
        logging.error('[Error] Number of classes is not the same. {}'.format(err))
        sys.exit()


    class_names = read_classes(FLAGS.classes)
    anchors = read_anchors(FLAGS.anchors)
    logging.info('Classes and Anchors loaded')

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)

    # 選一張圖像
    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)

    # 進行圖像輸入的前處理
    image = tf.image.resize(img_raw, (FLAGS.size, FLAGS.size)) # 修改輸入圖像大小來符合模型的要求
    image /= 255. # 進行圖像歸一處理
    image = tf.expand_dims(image, 0) # 增加 batch dimension

    # 進行圖像偵測
    yolo_outputs = yolo.predict(image)
    scores, boxes, classes = yolo_eval(
        yolo_outputs,
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
    image = img_raw.numpy()
    image = draw_outputs(image, (scores, boxes, classes), class_names, colors)

    # Show
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.show()

    # Save
    if FLAGS.save:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.output, image)


if __name__ == "__main__":
    app.run(main)
