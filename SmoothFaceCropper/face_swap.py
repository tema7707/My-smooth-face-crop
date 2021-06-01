import cv2
import numpy as np
import copy
import PIL.Image as Image
import sys
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('./FaceShifter/')
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from network.AEI_Net import *
from face_modules.mtcnn import *


class FaceSwapper():
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.G = AEI_Net(c_id=512)
        self.G.eval()
        self.G.load_state_dict(torch.load('./FaceShifter/saved_models/G_latest.pth', map_location=self.device))
        self.G = self.G.to(device)

        self.arcface = Backbone(50, 0.6, 'ir_se').to(self.device)
        self.arcface.eval()
        self.arcface.float()
        self.arcface.load_state_dict(torch.load('./FaceShifter/face_modules/model_ir_se50.pth', map_location=self.device), strict=False)

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        

    def generate(self, target, source):
        '''
        Generate face swap

        Parameters:
            target (np.array): Target face image
            source (np.array): The image that will be changed

        Returns:
            Yt (np.array): Generated face image
        '''
        Xs = cv2.resize(target.astype('float'), (256, 256), interpolation=cv2.INTER_CUBIC)
        Xt = cv2.resize(source.astype('float'), (256, 256), interpolation=cv2.INTER_CUBIC)

        Xs = self.test_transform(Xs / 255.)
        Xt = self.test_transform(Xt / 255.)

        Xs = Xs.unsqueeze(0).to(self.device)
        Xt = Xt.unsqueeze(0).to(self.device)

        Xs = Xs.float()
        Xt = Xt.float()
        with torch.no_grad():
            embeds, _ = self.arcface(F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True))
            embedt, __ = self.arcface(F.interpolate(Xt, (112, 112), mode='bilinear', align_corners=True))
            Yt, _ = self.G(Xt, embeds)
            Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5

        return Yt


def overlay(target, face, target_box):
    '''
    Overlay face image to target image.

    Parameters:
        target (np.array): The image where the overlay will take place
        face (np.array): The image that will be overlaid
        target_box (np.array): Region to overlay
            
    Returns:
        image (np.array): Overlaid image
    '''
    w, h = target_box[1]-target_box[0], target_box[3]-target_box[2]
    face = cv2.resize(face.astype('float'), (h, w), interpolation=cv2.INTER_CUBIC)
    image = copy.deepcopy(target)
    image[target_box[2]:target_box[3], target_box[0]:target_box[1]] = face
    return image
