import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

import yolov3.dataset as dataset
from yolov3.utils import read_classes
from yolov3.model import (
    yolo_anchors, yolo_anchor_masks
)

flags.DEFINE_string(
    'dataset', 'voc2012', 'The dataset that you want to visualize')
flags.DEFINE_enum(
    'split', 'train', ['train', 'val', 'test'], 'train or val or test dataset')
flags.DEFINE_integer(
    'n', 3, 'Number of images you want to show. (low_bound is 3)', lower_bound=3)


def visualize_raw_data(dataset,
                       class_names,
                       n = 3,
                       font_face = cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale = 1,
                       font_thickness = 2):
    """Visualize the dataset."""
    n = FLAGS.n
    col = 2
    if n % 2 != 0:
        row = (n // 2) + 1
    else:
        row = n // 2

    fig, axes = plt.subplots(row, col, figsize=(8, 8))
    fig.subplots_adjust(hspace = .5, wspace = .5)
    idx_row = idx_col = 0
    for x, y in dataset.take(FLAGS.n):
        img = x.numpy()
        height, width = tf.shape(img)[0].numpy(), tf.shape(img)[1].numpy()

        for xmin, ymin, xmax, ymax, label in y.numpy():
            left = (xmin * width).astype('int32')
            top = (ymin * height).astype('int32')
            right = (xmax * width).astype('int32')
            bottom = (ymax * height).astype('int32')
            label = class_names[int(label)]

            # cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img, '{} {:.4f}'.format(label, 1.0000), (left, int(top - 4)),
                        font_face, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

        if idx_col != 0 and idx_col % 2 == 0:
            idx_row += 1
            idx_col = 0

        axes[idx_row, idx_col].imshow(img)
        idx_col += 1

    plt.show()


def visualize_data(dataset, anchors):
    """Show the result of the dataset after preprocessing."""
    for x, y_outs in dataset.take(3):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))
        f.subplots_adjust(hspace = .2, wspace = .05)
        axs = (ax1, ax2, ax3)

        for anchor_box_idx, y in enumerate(y_outs):
            img = x.numpy()
            img = np.array(img * 255, dtype=np.uint8)
            true_boxes = y.numpy()

            # Custom (rgb) grid color and object color
            colors = [(255,0,0), (0,0,255), (255,255,0)]
            grid_color = [255, 255, 255] # (255,255,255)

            # Plot grid box
            # Modify the image to include the grid
            dowsample_size = 32 // pow(2, anchor_box_idx)
            dx, dy = (dowsample_size, dowsample_size) # downsamples the input by 32
            img[:,::dy,:] = grid_color
            img[::dx,:,:] = grid_color

            # Plot anchor box
            anchor_exist = tf.not_equal(true_boxes[:, :, :, 0], 0)
            anchor_boxes_idx = tf.cast(tf.where(anchor_exist), tf.int32)
            for dy, dx, anchor_idx in anchor_boxes_idx:
                # 1. anchor box center
                anchor_boxes_xy = [(dx * dowsample_size, dy * dowsample_size)]
                for i, box_xy in enumerate(anchor_boxes_xy):
                    cv2.circle(img, box_xy, 10, colors[anchor_idx], -1)

                # 2. anchor box
                anchor_box_wh = anchors[6 - anchor_box_idx * 3 + anchor_idx] * 416
                anchor_box_wh_half = anchor_box_wh / 2.
                bbox_mins = anchor_boxes_xy - anchor_box_wh_half
                bbox_maxes = anchor_boxes_xy + anchor_box_wh_half

                for i in range(len(bbox_mins)):
                    cv2.rectangle(img, (int(bbox_mins[i][0]), int(bbox_mins[i][1])), (int(bbox_maxes[i][0]), int(bbox_maxes[i][1])), colors[anchor_idx], 2)

            # Plot true box
            true_bbox = true_boxes[..., 0:4] * 416

            for grid_y in true_bbox:
                for grid_x in grid_y:
                    for box in grid_x:
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            axs[anchor_box_idx].imshow(img)
        f.tight_layout()

    plt.show()

def main(argv):
    # Load the tf.data.Dataset from TFRecord files
    dataset_path = os.path.join('./data', FLAGS.dataset + '_' + FLAGS.split + '.tfrecord')
    raw_dataset = dataset.load_tfrecord_dataset(dataset_path)

    # class_names
    classes_path = os.path.join('./model_data', FLAGS.dataset + '_classes.txt')
    class_names =  read_classes(classes_path)

    # Show images from the dataset
    visualize_raw_data(raw_dataset, class_names)

    # Preprocess the dataset
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks
    image_size = 416
    yolo_max_boxes = 100

    train_dataset = raw_dataset.map(lambda x, y: (
        dataset.preprocess_data(
            x, y,
            anchors, anchor_masks,
            image_size=image_size,
            yolo_max_boxes=yolo_max_boxes)
    ))

    visualize_data(train_dataset, anchors)


if __name__ == "__main__":
    app.run(main)
