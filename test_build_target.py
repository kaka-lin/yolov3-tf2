import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.dataset import preprocess_data

# anchor boxes
anchors = np.array([
    (10, 13), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326)], np.float32) / 416

anchor_masks = np.array([[6, 7, 8],
                         [3, 4, 5],
                         [0, 1, 2]])


def draw_boxes(x, y_outs):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 30))
    f.subplots_adjust(hspace = .2, wspace = .05)
    axs = (ax1, ax2, ax3)

    for anchor_box_idx, y in enumerate(y_outs):
        img = x.numpy()
        img = np.array(img * 255, dtype=np.uint8)
        true_boxes = y.numpy()

        # Custom (rgb) grid color and object color
        colors = [(255,0,0), (0,255,0), (0,0,255)]
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
            for i, (box_x, box_y) in enumerate(anchor_boxes_xy):
                box_x, box_y = box_x.numpy(), box_y.numpy()
                cv2.circle(img, (box_x, box_y), 10, colors[anchor_idx], -1)

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
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 50 * anchor_box_idx), 2)

        axs[anchor_box_idx].imshow(img)
    f.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 選一張圖像
    img_raw = tf.image.decode_image(
        open('data/horse.jpg', 'rb').read(), channels=3)

    true_boxes = tf.constant(
        [[0.106, 0.196832582, 0.942, 0.950226247, 12.],
         [0.316, 0.0995475128, 0.578, 0.377828062, 14.]])

    # Change true boxes to training labels
    x_train, y_train = preprocess_data(img_raw, true_boxes, anchors, anchor_masks)

    # draw results
    draw_boxes(x_train, y_train)
