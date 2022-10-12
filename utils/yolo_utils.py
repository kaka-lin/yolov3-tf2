import colorsys
import random

import cv2
import numpy as np
import tensorflow as tf

# anchor boxes
yolo_anchors = np.array([
    (10, 13), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326)], np.float32) / 416

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


def scale_boxes(boxes, inpute_shape, image_shape):
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
    scale = image_shape / inpute_shape
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

def yolo_boxes(pred, input_dims, anchors):
    # pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...classes))
    grid_size = tf.shape(pred)[1:3]

    box_xy = tf.sigmoid(pred[..., 0:2])
    box_wh = pred[..., 2:4]
    box_confidence = tf.sigmoid(pred[..., 4:5])
    box_class_probs = tf.sigmoid(pred[..., 5:])

    # box_xy: (grid_size, grid_size, num_anchors, 2)
    # grid: (grdid_siez, grid_size, 1, 2)
    #       -> [0,0],[0,1],...,[0,12],[1,0],[1,1],...,[12,12]
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    # stride: 416 / grid_size
    # [416/52, 416/26, 416/13] -> [8, 16, 32]
    stride = tf.cast(input_dims // grid_size, tf.float32)
    box_xy = (box_xy + tf.cast(grid, tf.float32)) * stride

    box_wh = tf.exp(box_wh) * anchors
    pred_box = tf.concat((box_xy, box_wh), axis=-1)
    return pred_box, box_confidence, box_class_probs


def yolo_boxes_and_scores(yolo_output, input_dims, anchors, classes):
    """Process output layer"""
    # yolo_boxes: pred_box, box_confidence, box_class_probs, pred_raw_box
    pred_box, box_confidence, box_class_probs = yolo_boxes(
        yolo_output, input_dims, anchors)

    # Convert boxes to be ready for filtering functions.
    # Convert YOLO box predicitions to bounding box corners.
    # (x, y, w, h) -> (x1, y1, x2, y2)
    box_xy = pred_box[..., 0:2]
    box_wh = pred_box[..., 2:4]

    box_x1y1 = box_xy - (box_wh / 2.)
    box_x2y2 = box_xy + (box_wh / 2.)
    boxes = tf.concat([box_x1y1, box_x2y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    # Compute box scores
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, classes])
    return boxes, box_scores


def yolo_non_max_suppression(boxes, box_scores,
                             classes=80,
                             max_boxes=100,
                             score_threshold=0.5,
                             iou_threshold=0.5):
    """Perform Score-filtering and Non-max suppression

    boxes: (10647, 4)
    box_scores: (10647, 80)
    # 10647 = (13*13 + 26*26 + 52*52) * 3(anchor)
    """

    # Create a mask, same dimension as box_scores.
    mask = box_scores >= score_threshold # (10647, 80)

    output_boxes = []
    output_scores = []
    output_classes = []

    # Perform NMS for all classes
    for c in range(classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        selected_indices = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes, iou_threshold)

        class_boxes = tf.gather(class_boxes, selected_indices)
        class_box_scores = tf.gather(class_box_scores, selected_indices)

        classes = tf.ones_like(class_box_scores, 'int32') * c

        output_boxes.append(class_boxes)
        output_scores.append(class_box_scores)
        output_classes.append(classes)

    output_boxes = tf.concat(output_boxes, axis=0)
    output_scores = tf.concat(output_scores, axis=0)
    output_classes = tf.concat(output_classes, axis=0)

    return output_scores, output_boxes, output_classes


# post-processing
def yolo_eval(yolo_outputs,
              anchors,
              image_shape, # (h, w)
              input_dims=(416, 416),
              letterbox=True,
              classes=80,
              max_boxes=100,
              score_threshold=0.5,
              iou_threshold=0.5):
    # Retrieve outputs of the YOLO model.
    num_layers = len(yolo_outputs)
    anchors = anchors
    anchor_mask = yolo_anchor_masks

    for i in range(0, num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_outputs[i],
            input_dims,
            anchors[anchor_mask[i]],
            classes)

        if i == 0:
            boxes, box_scores = _boxes, _box_scores
        else:
            boxes = tf.concat([boxes, _boxes], axis=0)
            box_scores = tf.concat([box_scores, _box_scores], axis=0)

    # Perform Score-filtering and Non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(boxes, box_scores,
                                                      classes,
                                                      max_boxes,
                                                      score_threshold,
                                                      iou_threshold)

    if letterbox:
        boxes = rescale_boxes(boxes, input_dims, image_shape) # letter
    else:
        boxes = scale_boxes(boxes, input_dims, image_shape) # resize
    return scores, boxes, classes


def draw_outputs(image, outputs, class_names, colors):
    image_shape = np.array((image.shape[1], image.shape[0]))
    inpute_shape = np.array((416, 416))
    h, w, _ = image.shape
    scores, boxes, classes = outputs
    #boxes = scale_boxes(boxes, (h, w)) # resize

    # for i in range(scores.shape[0]):
    #     x1, y1, x2, y2 = boxes[i]
    #     # min((416, 416) / (1242, 375)
    #     scale = min(inpute_shape / image_shape)
    #     resized_shape = image_shape * scale # (416.0, 125.60386473429952)
    #     pad_size = inpute_shape - resized_shape # (0, 290.3961352657005)
    #     x1 = int(((x1 - pad_size[0] // 2) / resized_shape[0]) * (image_shape[0]))
    #     y1 = int(((y1 - pad_size[1] // 2) / resized_shape[1]) * (image_shape[1]))
    #     x2 = int(((x2 - pad_size[0] // 2) / resized_shape[0]) * (image_shape[0]))
    #     y2 = int(((y2 - pad_size[1] // 2) / resized_shape[1]) * (image_shape[1]))
    #     print("Confidence: {}, Class: {}, Box: {}".format(
    #         scores[i], class_names[int(classes[i])], [x1, y1, x2, y2]))
    #     class_id = int(classes[i])
    #     predicted_class = class_names[class_id]
    #     score = scores[i].numpy()

    #     label = '{} {:.2f}'.format(predicted_class, score)

    #     # colors: RGB
    #     cv2.rectangle(image, (x1, y1), (x2, y2), tuple(colors[class_id]), 2)

    #     font_face = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1
    #     font_thickness = 2

    #     label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
    #     label_rect_left, label_rect_top = int(x1 - 3), int(y1 - 3)
    #     label_rect_right, label_rect_bottom = int(x1 + 3 + label_size[0]), int(y1 -  label_size[1])
    #     cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
    #                   tuple(colors[class_id]), -1)

    #     cv2.putText(image, label, (x1, int(y1 - 4)),
    #                 font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

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
