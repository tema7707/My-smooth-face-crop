import cv2
import numpy as np
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import math


def get_frame(video_path, frame_number=0):
    '''
    Returns the specified frame from the video.

    Parameters:
        video_path (str): Path to the video
        frame_number (int): Number of frame from video
            
    Returns:
        image (np.array): Specified frame from the video
    '''
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success, image = vidcap.read()
    while success and count < frame_number:
        success, image = vidcap.read()
        count += 1
        
    if count < frame_number:
        raise RuntimeError('Frame number is too big for this video.')
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_all_frames(video_path):
    '''
    Returns all frames from the video.

    Parameters:
        video_path (str): Path to the video
            
    Returns:
        frames (List[np.array]): All frames from the video
    '''
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    success, image = vidcap.read()
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
        success, image = vidcap.read()
        
    return frames


def write_video(file_path, frames, fps):
    '''
    Writes frames to an mp4 video file
    
    Parameters:
        file_path: Path to output video, must end with .mp4
        frames: List of PIL.Image objects
        fps: Desired frame rate
    '''

    w, h = frames[0].shape[:-1]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.write(frame)

    writer.release() 


def cv_2_image(image):
    '''
    Convert cv2 image to PIL image.

    Parameters:
        image: cv2 image
            
    Returns:
        im_pil: PIL.Image
    '''
    im_pil = Image.fromarray(image)
    return im_pil


def image_2_numpy(image):
    '''
    Convert PIL image to np.array.

    Parameters:
        image: PIL image
            
    Returns:
        image: np.array
    '''
    return np.asarray(image)


def unit_vector(vector):
    ''' Returns the unit vector of the vector '''
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    ''' Returns the angle in radians between vectors '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotate_image(image, angle):
    row, col = image.shape[:-1]
    center = tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def rad_to_angel(rad):
    return rad / (2 * math.pi) * 360
