from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

from utils.common import *
from utils.yolo_utils import yolo_anchors, yolo_anchor_masks
import utils.dataset as dataset
from models.yolov3 import (
    Yolov3, YoloLoss, freeze_all
)

flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 10, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('num_classes', 20, 'number of classes in the model')
flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_string('train_dataset', './data/voc2012_train.tfrecord',
                    'path to the train dataset')
flags.DEFINE_string('val_dataset', './data/voc2012_val.tfrecord',
                    'path to the validation dataset')
flags.DEFINE_boolean('transfer', True, 'Transfer learning or not')
flags.DEFINE_string('pretrained_weights', './checkpoints/yolov3.tf',
                    'path to prttrained weights file')
flags.DEFINE_integer('weights_num_classes', 80,
                     'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_string('output', './model_data/yolov3.h5',
                    'path to save model')


def train(argv):
    # Load the tf.data.Dataset from TFRecord files
    raw_train_ds = dataset.load_tfrecord_dataset(FLAGS.train_dataset)
    raw_val_ds = dataset.load_tfrecord_dataset(FLAGS.val_dataset)

    # Preprocess the dataset
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    train_ds = raw_train_ds.map(lambda x, y: (
        dataset.preprocess_data(
            x, y,
            anchors, anchor_masks,
            image_size=FLAGS.size,
            yolo_max_boxes=FLAGS.yolo_max_boxes)
    ))
    train_ds = train_ds.shuffle(buffer_size=512).batch(FLAGS.batch_size)
    train_ds = train_ds.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_ds = raw_val_ds.map(lambda x, y: (
        dataset.preprocess_data(
            x, y,
            anchors, anchor_masks,
            image_size=FLAGS.size,
            yolo_max_boxes=FLAGS.yolo_max_boxes)
    ))
    val_ds = val_ds.batch(FLAGS.batch_size)

    # Build the model
    model = Yolov3(FLAGS.size, classes=FLAGS.num_classes)
    #model.summary()

    # Configure the model for transfer learning
    if FLAGS.transfer:
        logging.info(">>> Transfer Learning: Darknet")

        # pretrained model
        model_pretrained = Yolov3(FLAGS.size, classes=FLAGS.weights_num_classes)
        model_pretrained.load_weights(FLAGS.pretrained_weights)

        model.get_layer('yolo_darknet').set_weights(
            model_pretrained.get_layer('yolo_darknet').get_weights())

        # freeze darknet and fine tune other layers
        freeze_all(model.get_layer('yolo_darknet'))
    else:
        logging.info(">>> Training from scratch")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    yolo_loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
                 for mask in anchor_masks]

    model.compile(optimizer=optimizer,
                  loss=yolo_loss,
                  run_eagerly=False)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    # Train the model
    history = model.fit(train_ds,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=val_ds)

    # Save the model
    model.save("./model_data/yolov3.h5")

if __name__ == "__main__":
    app.run(train)
