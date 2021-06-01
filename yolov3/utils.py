import cv2
import numpy as np
import tensorflow as tf

# For preprocess
def letterbox_image(image, input_dims):
    """resize image with unchanged aspect ratio using padding"""
    h, w, _ = image.shape
    desired_w, desired_h = input_dims # (416, 416)
    scale = min(desired_w/w, desired_h/h)

    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

    padding_image = np.ones((desired_h, desired_w, 3), np.uint8) * 128
    # Put the image that after resized into the center of new image
    # 將縮放後的圖片放入新圖片的正中央
    h_start = (desired_h - new_h) // 2
    w_start = (desired_w - new_w) // 2
    padding_image[h_start:h_start+new_h, w_start:w_start+new_w, :] = image

    return padding_image


# For postprocess
def yolo_correct_boxes(boxes, image_shape, input_dims):
    """Coordinate transformation

    Because our output tensor are predictions on the `padding image`,
    and not the original image, we need to transform the coord
    from relative the padding image to relative the original image (after resize)

    for exmaple:
    orginal_image: (576, 768) -> (416, 416)
    scale_size: min((416/576), (416/768)) = 0.54167
    resized_image: (312, 416)
    padding_image: (416. 416)

    => the (x, y) point in `padding_image`
    need to trandform to in `resized_image`

    => resize_y = (padding_y - scale * origin_h) / 2

    ### Note: (scale * origin_h) / 2 or (scale * origin_w) / 2
    -> the shift between `resized_image` and `padding_image`
    """
    h, w = image_shape
    desired_w, desired_h = input_dims
    scale = min(desired_w/w, desired_h/h)

    offset_x = (desired_w - scale * w) / 2. / desired_w
    offset_y = (desired_h - scale * h) / 2 / desired_h
    offsets = [offset_x, offset_y, offset_x, offset_y]

    boxes = (boxes - offsets)

    return boxes, scale


