import face_alignment
import PIL.Image
import copy
import math
import numpy as np
from skimage import io
from SmoothFaceCropper import utils


def crop(image, detector, padding=0, with_rotation=False):
    '''
    Returns the cropped face and its borders.

    Parameters:
            image (np.array): Source image
            detector (FaceAlignment): FaceAlignment detector for landmarks localization 

    Returns:
            face (np.array), borders (np.array): Cropped face ans its borders
    '''
    if with_rotation:
        image = rotate(image, detector)
        
    preds = detector.get_landmarks(image)
        
    # найдем крайние точки
    x_min = int(min(preds[0][:, 0]))
    x_max = int(max(preds[0][:, 0]))
    y_min = int(min(preds[0][:, 1]))
    y_max = int(max(preds[0][:, 1]))

    # достроим до квадрата
    w = x_max - x_min
    h = y_max - y_min
    if w > h:
        y_max += (w - h) // 2
        y_min -= math.ceil((w - h) / 2)
    else:
        x_max += (h - w) // 2
        x_min -= math.ceil((h - w) / 2)
      
    padding_x = int((x_max - x_min) * padding)
    padding_y = int((y_max - y_min) * padding)

    x_current_min = x_min - padding_x
    x_current_max = x_max + padding_x
    y_current_min = y_min - padding_y
    y_current_max = y_max + padding_y

    # обработаем выходы за границы
    l_x_pad, r_x_pad, u_y_pad, d_y_pad = 0, 0, 0, 0
    if x_current_min < 0:
        l_x_pad = -x_current_min
        x_current_min = 0
        x_current_max += l_x_pad
    if x_current_max > image.shape[1]:
        r_x_pad = x_current_max - image.shape[1]
    if y_current_min < 0:
        u_y_pad = -y_current_min
        y_current_min = 0
        y_current_max += u_y_pad
    if y_current_max > image.shape[0]:
        d_y_pad = y_current_max - image.shape[0]

    current_image = np.pad(array=image, pad_width=((u_y_pad, d_y_pad), (l_x_pad, r_x_pad), (0, 0)), mode='constant', constant_values=0)
    crop_img = current_image[y_current_min:y_current_max, x_current_min:x_current_max]
    assert crop_img.shape[0] == crop_img.shape[1] 

    return crop_img, (x_current_min, x_current_max, y_current_min, y_current_max)


def rotate(image, detector):
    '''
    Rotate image to centralize
    '''
    preds = detector.get_landmarks(image)
    detection = preds[0]
    A = np.array([detection[30,0], detection[30,1]])
    B = np.array([detection[8,0], detection[8,1]])
    C = np.array([detection[8,0], detection[30,1]])
    A = A - B
    B = C - B
    angle = utils.rad_to_angel(utils.angle_between(A, B))
    if A[0] < 0:
        angle = 360 - angle
    image = utils.rotate_image(image, angle)
    return image
