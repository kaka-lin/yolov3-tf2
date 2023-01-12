import colorsys
import random

import cv2
import numpy as np
import tensorflow as tf

# anchor boxes
yolo_anchor_masks = np.array([[6, 7, 8],
                              [3, 4, 5],
                              [0, 1, 2]])


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def scale_boxes(boxes, input_shape, image_shape):
    """ Rescales bounding boxes to the original image shape

    This is for cv::resize().

    Args:
        inpu_shape: model input shape. (h, w)
        image_shape: the shape of origin image. (h, w)
    """
    image_shape = np.array(image_shape)

    # Example:
    #   scale: (1200, 630) / (416, 416) = (2.88461538, 1.51442308)
    #   resize_shape: (416, 416) unpad_size
    scale = image_shape / input_shape
    scale_dims = tf.stack([scale[1], scale[0], scale[1], scale[0]])
    scale_dims = tf.cast(tf.reshape(scale_dims, [1, 4]), tf.float32)
    boxes = boxes * scale_dims
    return boxes


def rescale_boxes(boxes, inpute_shape, image_shape):
    """ Rescales bounding boxes to the original image shape.

    This is for letterbox().

    Args:
        inpu_shape: model input shape. (h, w)
        image_shape: the shape of origin image. (h, w)
    """
    height, width = image_shape[0], image_shape[1]
    image_shape = np.array((width, height))

    # 1. Calculate padding_size
    #
    # The amount of padding that was added
    # Example:
    #   min((416, 416) / (1200, 630), (w, h)
    #   resized_shape: (416.0, 218.4), unpad_size
    #   pad_size:      (0,     197.6)
    scale = min(inpute_shape / image_shape)
    resized_shape = image_shape * scale # unpad_size
    pad_size = inpute_shape - resized_shape

    # For tf 2.x
    pad_size = tf.stack([pad_size[0], pad_size[1], pad_size[0], pad_size[1]])
    pad_size = tf.cast(tf.reshape(pad_size, [1, 4]), tf.float32)
    resized_shape = tf.stack([resized_shape[0], resized_shape[1], resized_shape[0], resized_shape[1]])
    resized_shape = tf.cast(tf.reshape(resized_shape, [1, 4]), tf.float32)

    image_shape = tf.stack([width, height, width, height])
    image_shape = tf.cast(tf.reshape(image_shape, [1, 4]), tf.float32)

    # 2. Rescale bounding boxes to dimension of original image
    # 與 scale_boxes() 一樣，如下：
    #   ((boxes - pad_size // 2) => new_boxes: the boxes position on resize_image
    #   (new_boxes / resized_shape) * image_shape
    #   = new_boxes * image_shape / resize_shape (in scale_boxes() is inpute_shape)
    boxes = ((boxes - pad_size // 2) / resized_shape) * image_shape
    return boxes


###########################################################################################


def broadcast_iou(box1, box2):
    """ Calculate Final IOU

    An efficient way to calculate the IOU matrix.

    Args:
      box1: pred box, [N, 13, 13, 3, (x1, y1, x2, y2)]
      box2: ground truth box, [V, (x1, y1, x2, y2)]

    Returns:
      [batch_size, grid, grid, anchors, num_gt_box]: [N, 13, 13, 3, V]
    """
    # box1: (..., (x1, y1, x2, y2))
    # box2: (V, (x1, y1, x2, y2))

    # broadcast boxes
    box1 = tf.expand_dims(box1, -2) # (..., 1, (x1, y1, x2, y2))
    box2 = tf.expand_dims(box2, 0)  # (1, V, (x1, y1, x2, y2))

    # new_shape: (..., V, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box1), tf.shape(box2))
    box1 = tf.broadcast_to(box1, new_shape)
    box2 = tf.broadcast_to(box2, new_shape)

    # intersection area
    # shape: [N, 13, 13, 3, V]
    inter_w = tf.maximum(tf.minimum(box1[..., 2], box2[..., 2]) -
                       tf.maximum(box1[..., 0], box2[..., 0]), 0)
    inter_h = tf.maximum(tf.minimum(box1[..., 3], box2[..., 3]) -
                       tf.maximum(box1[..., 1], box2[..., 1]), 0)
    inter_area = inter_w * inter_h

    # box area
    # shape: [N, 13, 13, 3, V]
    box1_area = (box1[..., 2] - box1[..., 0]) * \
        (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * \
        (box2[..., 3] - box2[..., 1])

    return inter_area / (box1_area + box2_area - inter_area)


def draw_outputs(image, outputs, class_names, colors):
    h, w, _ = image.shape
    scores, boxes, classes = outputs

    for i in range(scores.shape[0]):
        left, top, right, bottom = boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        class_id = int(classes[i])
        predicted_class = class_names[class_id]
        score = scores[i].numpy()

        label = '{} {:.2f}'.format(predicted_class, score)

        # colors: RGB
        cv2.rectangle(image, (left, top), (right, bottom), tuple(colors[class_id]), 2)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top -  label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
                      tuple(colors[class_id]), -1)

        cv2.putText(image, label, (left, int(top - 4)),
                    font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return image
