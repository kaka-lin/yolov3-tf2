import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3.model import Yolov3
from yolov3.utils import load_darknet_weights

flags.DEFINE_string('weights', './model_data/yolov3.weights',
                    'path to input weights file (darknet)')
flags.DEFINE_integer('weights_num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('output', './checkpoints/yolov3.tf',
                    'path to output weights file (tensorflow)')


# Convert Darknet weights to Tensorflow weights
def main(argv):
    yolo = Yolov3(classes=FLAGS.weights_num_classes)
    yolo.summary()
    logging.info('Model created')

    load_darknet_weights(yolo, FLAGS.weights)
    logging.info('Weights loaded')

    yolo.save_weights(FLAGS.output)
    logging.info('Weights saved')


if __name__ == "__main__":
    app.run(main)
