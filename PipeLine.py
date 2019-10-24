''' This file is YingLi's request
Input:  two image (not-aligned)
Output: similar score
=========requirement ============
1: face detection & alignment
'''

#====================================== Dependent file ===================================
import os,sys
import dlib
import cv2
import numpy as np
import model_define as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
# model: detection & align
detector = dlib.get_frontal_face_detector()                         # traditional detector
cnn_face_detector = dlib.cnn_face_detection_model_v1('./model/mmod_human_face_detector.dat')   # cnn detector
sp = dlib.shape_predictor('./model/shape_predictor_5_face_landmarks.dat')                           #

# model: feature extraction
resnet18 = './model/resnet18.tar'
#========================================= Function =======================================
def face_detection_alignmen(original_image):
    # Load the image using OpenCV
    bgr_img = cv2.imread(original_image)

    # Convert to RGB since dlib uses RGB images
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # first traditional detector for efficient
    dets = detector(img, 1)
    num_faces = len(dets)

    # If there is no face detected in the image
    if num_faces == 0:
        # second use cnn detector
        cnn_dets = cnn_face_detector(img, 1)
        dets = dlib.rectangles()
        dets.extend([d.rect for d in cnn_dets])
        num_faces = len(dets)
        if num_faces == 0:
            print('There is no face in the image')

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    image = dlib.get_face_chip(img, faces[0], size=224, padding=0.3)
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('annotShow', cv_bgr_img)
    # cv2.waitKey(0)

    return cv_bgr_img

def load_resnet18():

    _structure = models.resnet18(num_classes=8)
    pretrained_state_dict = torch.load(resnet18)['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if '.num_batches_tracked' in key:
            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model

def expression_feature_extraction(aligned_face):


    # Load model
    model = load_resnet18()
    # numpy2tensor
    transform = transforms.ToTensor()
    tensor_face = transform(aligned_face).unsqueeze(0)
    # feature
    feature, softmax_score = model(tensor_face)

    return feature, softmax_score

def measure_similiar(feature1, feature2):
    feature1 = F.normalize(feature1)
    feature2 = F.normalize(feature2)
    distance = feature1.mm(feature2.t())

    return distance


#========================================== Main Function ===================================
def main(cartoon_dir, realman_dir):

    # image preprocess
    aligned_cartoon = face_detection_alignmen(cartoon_dir)
    aligned_realman = face_detection_alignmen(realman_dir)

    # feature extraction
    feature_cartoon, score_cartoon = expression_feature_extraction(aligned_cartoon)
    feature_realman, score_realman = expression_feature_extraction(aligned_realman)

    # similiar score

    feature_similiar = measure_similiar(feature_cartoon, feature_realman)
    score_similiar   = measure_similiar(score_cartoon, score_realman)
    return feature_similiar, score_similiar

if __name__ == 'main':
    cartoon_dir = '/home/dbmeng/LibMeng/Database/Emotion/EmotiW19/FrameVal/Fear/000405847.avi/0000009.png'
    realman_dir = '/home/dbmeng/LibMeng/Database/Emotion/EmotiW19/FrameVal/Happy/000403327.avi/0000010.png'
    feature_similiar, score_similiar = main(cartoon_dir, realman_dir)