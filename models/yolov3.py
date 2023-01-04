import numpy as np
from absl import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    LeakyReLU,
    MaxPool2D,
    Lambda,
    concatenate,
    Add,
    ZeroPadding2D,
    UpSampling2D,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

from utils.yolo_utils import (
    yolo_anchors, yolo_anchor_masks,
    broadcast_iou,
    scale_boxes,
    rescale_boxes
)

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def load_darknet_weights(model, weights_file):
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        layers = YOLOV3_LAYER_LIST

        for layer_name in layers:
            sub_model = model.get_layer(layer_name)
            for i, layer in enumerate(sub_model.layers):
                if not layer.name.startswith('conv2d'):
                    continue

                # BatchNormalization layer
                batch_norm = None
                if i + 1 < len(sub_model.layers) and \
                        sub_model.layers[i + 1].name.startswith('batch_norm'):
                    batch_norm = sub_model.layers[i + 1]

                logging.info("{}/{} {}".format(
                    sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

                filters = layer.filters
                kerner_size = layer.kernel_size[0]
                #input_dim = layer.get_input_shape_at(0)[-1]
                input_dim = layer.input_shape[-1]

                if batch_norm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                else:
                    # darknet [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(
                        wf, dtype=np.float32, count=4*filters)
                    # tf [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                # darknet shape (out_dim, input_dim, height, width)
                conv_shape = (filters, input_dim, kerner_size, kerner_size)
                conv_weights = np.fromfile(
                    wf, dtype=np.float32, count=np.product(conv_shape))

                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0])

                if batch_norm is None:
                    layer.set_weights([conv_weights, conv_bias])
                else:
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)

            logging.info("Completed!")
    logging.info("Weights loaded!")

# NOTE: would cause NAN during inference
#   reference: 1. https://github.com/keras-team/keras/issues/17204
#              2. https://github.com/zzh8829/yolov3-tf2/pull/188
#   -> We using official tf.keras.layers.BatchNormalization
#
# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     """
#     Make trainable=False freeze BN for real (the og version is sad)
#     """

#     def call(self, x, training=False):
#         if training is None:
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)


def DarknetConv2D(x, filters, size, stride=1, batch_norm=True):
    if stride == 1:
        padding = 'same'
    else:
        # dowsample: top left half-padding
        # padding=((top, bottom), (left, right))
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'

    x = Conv2D(filters=filters, kernel_size=size,
               strides=(stride, stride), padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv2D(x, filters // 2, 1)
    x = DarknetConv2D(x, filters, 3)
    x = Add()([prev, x])
    return x


# ResidualBlock
def DarknetBlock(x, filters, num_blocks):
    x = DarknetConv2D(x, filters, 3, stride=2)
    for _ in range(num_blocks):
        x = DarknetResidual(x, filters)
    return x


def darknet_body(name=None):
    x = inputs = Input([None, None, 3])

    # Darknet53
    x = DarknetConv2D(x, 32, 3)
    x = DarknetBlock(x, 64, num_blocks=1)
    x = DarknetBlock(x, 128, num_blocks=2)
    x = x_36 = DarknetBlock(x, 256, num_blocks=8) # skip connection
    x = x_61 = DarknetBlock(x, 512, num_blocks=8) # conv + residual
    x = DarknetBlock(x, 1024, num_blocks=4) # x_74

    return Model(inputs, (x_36, x_61, x), name=name)


def yolo_body(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv2D(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = concatenate([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv2D(x, filters, 1)
        x = DarknetConv2D(x, filters * 2, 3)
        x = DarknetConv2D(x, filters, 1)
        x = DarknetConv2D(x, filters * 2, 3)
        x = DarknetConv2D(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def yolo_output(filters, num_anchors, classes, name=None):
    def yolo_output_conv(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv2D(x, filters * 2, 3)
        x = DarknetConv2D(x, (num_anchors * (classes + 5)), 1, batch_norm=False)
        # output
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            num_anchors, classes + 5)))(x)
        return Model(inputs, x, name=name)(x_in)
    return yolo_output_conv


def yolo_boxes(pred, input_dims, anchors, isTraining=False):
    # pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...classes))
    grid_size = tf.shape(pred)[1:3]

    box_xy = tf.sigmoid(pred[..., 0:2])
    box_wh = pred[..., 2:4]
    box_confidence = tf.sigmoid(pred[..., 4:5])
    box_class_probs = tf.sigmoid(pred[..., 5:])

    # original `xywh` for loss
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    # box_xy: (grid_size, grid_size, num_anchors, 2)
    # grid: (grdid_siez, grid_size, 1, 2)
    #       -> [0,0],[0,1],...,[0,12],[1,0],[1,1],...,[12,12]
    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    if not isTraining:
        # stride: 416 / grid_size
        # [416/52, 416/26, 416/13] -> [8, 16, 32]
        stride = tf.cast(input_dims // grid_size, tf.float32)
        box_xy = (box_xy + tf.cast(grid, tf.float32)) * stride
        box_wh = tf.exp(box_wh) * anchors
    else:
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - (box_wh / 2.)
    box_x2y2 = box_xy + (box_wh / 2.)
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, box_confidence, box_class_probs, pred_box


def yolo_boxes_and_scores(yolo_output, input_dims, anchors, classes):
    """Process output layer"""
    # yolo_boxes: pred_box, box_confidence, box_class_probs, pred_raw_box
    pred_box, box_confidence, box_class_probs, pred_xywh = yolo_boxes(
        yolo_output, input_dims, anchors)

    # Reshape box to: [N, (x1, y1, x2, y2)]
    boxes = tf.reshape(pred_box, [-1, 4])

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


def Yolov3(size=None, channels=3, masks=yolo_anchor_masks,
           classes=80):
    x = inputs = Input([size, size, channels], name='input')

    # Darknet53

    x_36, x_61, x = darknet_body(name='yolo_darknet')(x) # x_74

    ##############################################################################
    # Yolo Body
    x = yolo_body(512, name='yolo_conv_0')(x) # x_79
    # Yolo Output 1. 13x13x(anchor*(classes+5)
    output_0 = yolo_output(512, len(masks[0]), classes, name='yolo_output_0')(x) # x_82

    # 82, output_0
    # 83, route -4 -> x_79
    # x_79 upsample concate x_61 && Yolo Body
    x = yolo_body(256, name='yolo_conv_1')((x, x_61)) # x_91
    # Yolo Output 2. 26x26x(anchor*(classes+5)
    output_1 = yolo_output(256, len(masks[1]), classes, name='yolo_output_1')(x) # x _94

    # 94, output_1
    # 95. route -4 -> x_91
    # x_91 upsample + x_36 && Yolo Body
    x = yolo_body(128, name='yolo_conv_2')((x, x_36)) # x_103
    # Yolo Output 3. 52x52x(anchor*(classes+5)
    output_2 = yolo_output(128, len(masks[2]), classes, name='yolo_output_2')(x) # x_106

    return Model(inputs, (output_0, output_1, output_2), name='yolov3')


###########################################################################################
# For training

def YoloLoss(anchors, input_dims=(416, 416), classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        """
        pred_box:  [batch_size, grid, grid, anchors, (x1, y1, x2, y2)]
        pred_xywh: [batch_size, grid, grid, anchors, (tx, ty, tw, th)]
        true_box:  [batch_size, grid, grid, anchors, (x1, y1, x2, y2)]
        """

        # 1. transform all pred outputs.
        # y_pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...cls))
        pred_box, pred_confidence, pred_class_probs, pred_xywh = yolo_boxes(
            y_pred, input_dims, anchors, isTraining=True)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs.
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, conf, cls))
        true_box, true_confidence, true_class = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. Invert ture_boxes to darknet style box to calculate loss.
        # true_box: already normalization to 0~1 through divided 416
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)

        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)  # avoid log(0)=-inf

        # 4. calculate all masks.
        #
        # 4-1. object mask: remove noobject cell.
        # true_confidence: cell has object or not
        #                 0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        # true_confidence_mask: [conv_height, conv_width, num_anchors]
        true_conf_mask = tf.squeeze(true_confidence, -1)

        # 4-2. ignore mask.
        #
        #  1. Find the IOU score of each predicted box with each ground truth box.
        #  2. Find the Best IOU scores.
        #  3. A confidence detector that this cell has an object
        #     if IOU > threshold otherwise no object.
        #
        # Description:
        #   - Reshape true_box: tf.boolean_mask(true_box, tf.cast(true_conf_mask, tf.bool))
        #                       (N, grid, grid, num_anchors, (x1, y1, x2, y2))
        #                    -> (N, num_true_boxes, 4)
        #   - best_iou: tf.reduce_max(iou, axis=-1)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, true_conf_mask),
            tf.float32)
        # ignore false positive when iou is over threshold
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses.
        #
        # 5-1. Calculate `coordinate loss`.
        xy_loss = box_loss_scale * true_conf_mask * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = box_loss_scale * true_conf_mask * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        # 5-2. Calculate `classification loss`.
        #
        # square(one_hot(true_class) - pred_class_probs)
        # TODO: use binary_crossentropy instead
        #   - true_class:       13x13x3x1
        #   - pred_class_probs: 13x13x3x20
        classification_loss = true_conf_mask * sparse_categorical_crossentropy(
                true_class, pred_class_probs)

        # 5-3. Calculate `Confidence loss`.
        objects_loss = binary_crossentropy(true_confidence, pred_confidence)
        confidence_loss = true_conf_mask * objects_loss + \
                          (1 - true_conf_mask) * ignore_mask * objects_loss

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1).
        xy_loss_sum = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss_sum = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        confidence_loss_sum = tf.reduce_sum(confidence_loss, axis=(1, 2, 3))
        classification_loss_sum = tf.reduce_sum(classification_loss, axis=(1, 2, 3))

        '''
        tf.print(xy_loss_sum)
        tf.print(wh_loss_sum)
        tf.print(confidence_loss_sum)
        tf.print(classification_loss_sum)
        '''

        return (xy_loss_sum + wh_loss_sum + confidence_loss_sum + classification_loss_sum)
    return yolo_loss


###########################################################################################
# For inference

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

if __name__ == "__main__":
    model = Yolov3(416, classes=20)
    model.summary()
