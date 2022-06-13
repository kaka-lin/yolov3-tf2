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

from utils.yolo_utils import yolo_anchors, yolo_anchor_masks

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


def Yolov3(size=None, channels=3,
           anchors=yolo_anchors, masks=yolo_anchor_masks,
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


def yolo_boxes(pred, anchors, classes):
    """YOLO bounding box formula

    bx = sigmoid(tx) + cx
    by = sigmoid(ty) + cy
    bw = pw * exp^(tw)
    bh = ph * exp^(th)
    Pr(obj) * IOU(b, object) = sigmoid(to) # confidence

    (tx, ty, tw, th, to) are the output of the model.
    """
    # pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...classes))
    grid_size = tf.shape(pred)[1]

    box_xy = tf.sigmoid(pred[..., 0:2])
    box_wh = pred[..., 2:4]
    box_confidence = tf.sigmoid(pred[..., 4:5])
    box_class_probs = tf.sigmoid(pred[..., 5:])

    # Darknet raw box
    pred_raw_box = tf.concat((box_xy, box_wh), axis=-1)

    # box_xy: (grid_size, grid_size, num_anchors, 2)
    # grid: (grdid_siez, grid_size, 1, 2)
    #       -> [0,0],[0,1],...,[0,12],[1,0],[1,1],...,[12,12]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    # stride = tf.cast(416 // grid_size, tf.float32)
    # box_xy = (box_xy + tf.cast(grid, tf.float32)) * stride
    box_wh = tf.exp(box_wh) * anchors
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    return pred_box, box_confidence, box_class_probs, pred_raw_box


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        """"1. transform all pred outputs."""
        # y_pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...cls))
        pred_box, pred_confidence, pred_class_probs, pred_raw_box = yolo_boxes(
            y_pred, anchors, classes)
        pred_raw_xy = pred_raw_box[..., 0:2]
        pred_raw_wh = pred_raw_box[..., 2:4]

        """2. transform all true outputs."""
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, conf, cls))
        true_box, true_confidence, true_class = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        true_boxes = tf.concat([true_xy, true_wh], axis=-1)

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        """3. Invert ture_boxes to darknet style box to calculate loss."""
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_raw_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_raw_wh = tf.math.log(true_wh / anchors)
        true_raw_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)  # avoid log(0)=-inf

        """4. calculate all masks."""
        """4-1. object mask: remove noobject cell."""
        # true_confidence: cell has object or not
        #                 0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        # true_confidence_mask: [conv_height, conv_width, num_anchors]
        true_conf_mask = tf.squeeze(true_confidence, -1)

        """4-2. ignore mask.

        1. Find the IOU score of each predicted box
           with each ground truth box.
        2. Find the Best IOU scores.
        3. A confidence detector that this cell has object
           if IOU > threshold otherwise no object.
        """
        # Reshape true_box: (N, grid, grid, num_anchors, 4) to (N, num_true_boxes, 4)
        true_boxes_flat = tf.boolean_mask(true_boxes, tf.cast(true_conf_mask, tf.bool))

        # broadcast shape: (N, grid, grid, num_anchors, num_true_boxes, (x, y, w, h))
        true_boxes = tf.expand_dims(true_boxes, -2) # (N, 13, 13, 3, 1, 4)
        true_boxes_flat = tf.expand_dims(true_boxes_flat, 0) # (1, num_true_boxes, 4)
        new_shape = tf.broadcast_dynamic_shape(tf.shape(true_boxes), tf.shape(true_boxes_flat)) # (N, 13, 13, 3, num_true_boxes, 4)

        # reshape: (batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params)
        true_boxes = tf.broadcast_to(true_boxes, new_shape)
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4] # (N, 13, 13, 5, num_true_boxes, 2)

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        # Expand pred (x,y,w,h) to allow comparison with ground truth.
        # (batch, conv_height, conv_width, num_anchors, 1, box_params)
        pred_xy = pred_box[..., 0:2]
        pred_wh = pred_box[..., 2:4]
        pred_xy = tf.expand_dims(pred_xy, 4)
        pred_wh = tf.expand_dims(pred_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersection_mins = tf.maximum(pred_mins, true_mins)
        intersection_maxes = tf.minimum(pred_maxes, true_maxes)
        intersection_wh = tf.maximum(intersection_maxes - intersection_mins, 0.)
        intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1] # (-1, 13, 13, 3, num_true_boxes)

        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        """4-2-1. Calculate IOU scores for each location."""
        union_areas = pred_areas + true_areas - intersection_areas
        iou_scores = intersection_areas / union_areas # (-1, 13, 13, 3, num_true_boxes)

        """4-2-2. Best IOU scores."""
        best_ious = tf.reduce_max(iou_scores, axis=4)

        """4-2-3. Ignore mask."""
        # ignore false positive when iou is over threshold
        ignore_mask = tf.cast(best_ious < ignore_thresh, tf.float32) # (-1, 13, 13, 3, 1)

        """5. calculate all losses."""
        # Calculate `coordinate loss`."""
        xy_loss = box_loss_scale * true_conf_mask * \
                  tf.reduce_sum(tf.square(true_raw_xy - pred_raw_xy), axis=-1)
        wh_loss = box_loss_scale * true_conf_mask * \
                  tf.reduce_sum(tf.square(true_raw_wh - pred_raw_wh), axis=-1)

        # Calculate `classification loss`."""
        # square(one_hot(true_class) - pred_class_probs)
        # TODO: use binary_crossentropy instead
        #   - true_class:       13x13x3x1
        #   - pred_class_probs: 13x13x3x20
        classification_loss = true_conf_mask * sparse_categorical_crossentropy(
                true_class, pred_class_probs)

        # Calculate Confidence loss."""
        objects_loss = binary_crossentropy(true_confidence, pred_confidence)
        confidence_loss = true_conf_mask * objects_loss + \
                          (1 - true_conf_mask) * ignore_mask * objects_loss

        """6. sum over (batch, gridx, gridy, anchors) => (batch, 1)."""
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


if __name__ == "__main__":
    model = Yolov3(416, classes=20)
    model.summary()
