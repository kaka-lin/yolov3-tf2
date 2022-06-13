import cv2
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


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


def preprocess_image(img, input_shape, 
                     model_type="tf", keep_aspect_ratio=True):
    """pre-process for yolo"""
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if keep_aspect_ratio:
        image = letterbox_image(img_rgb, input_shape)
    else:
        image = cv2.resize(img_rgb, input_shape)
    image = np.array(image, dtype='float32')
    image = image / 255.
    if model_type == "torch":
        image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return image
