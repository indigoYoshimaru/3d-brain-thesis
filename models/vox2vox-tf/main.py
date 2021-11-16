#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser('Vox2Vox training and validation script', add_help=False)

## training parameters
parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
parser.add_argument('-nc', '--num_classes', default=4, type=int, help='number of classes')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help='batch size')
parser.add_argument('-a', '--alpha', default=5, type=int, help='alpha weight')
parser.add_argument('-ne', '--num_epochs', default=200, type=int, help='number of epochs')

args = parser.parse_args()
gpu = args.gpu
n_classes = args.num_classes
batch_size = args.batch_size
alpha = args.alpha
n_epochs = args.num_epochs


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import nibabel as nib
import glob
import time
from tensorflow.keras.utils import to_categorical
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from scipy.ndimage.interpolation import affine_transform
from sklearn.model_selection import train_test_split

from utils import *
from augmentation import *
from losses import *
from models import *
from train_v2v import *

Nclasses = 4
classes = np.arange(Nclasses)

def read_images(img_dir:str)-> list:
    # images lists
    t1_list = sorted(glob.glob('{}/**/*t1.nii.gz'.format(img_dir), recursive=True))
    t2_list = sorted(glob.glob('{}/**/*t2.nii.gz'.format(img_dir), recursive=True))
    t1ce_list = sorted(glob.glob('{}/**/*t1ce.nii.gz'.format(img_dir), recursive=True))
    flair_list = sorted(glob.glob('{}/**/*flair.nii.gz'.format(img_dir), recursive=True))
    seg_list = sorted(glob.glob('{}/**/*seg.nii.gz'.format(img_dir)))

    Nim = len(t1_list)
    img_list =[]
    print(t1_list)
    for i in range(0,Nim): 
        img_list.append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
    return img_list

# create the training and validation sets

# idx = np.arange(Nim)
# idxTrain, idxValid = train_test_split(idx, test_size=0.25)
sets = {'train': [], 'valid': []}

# for i in idxTrain:
#     sets['train'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
# for i in idxValid:
#     sets['valid'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])

sets['train']=read_images('/content/drive/MyDrive/thesis/brain_segmentation/dataset/train')
sets['valid']=read_images('/content/drive/MyDrive/thesis/brain_segmentation/dataset/val')
# file_dir = '/mnt/d/2122sem1/pre_thesis/brain-reconstruction/code/read_dataset/data'
# sets['train']=read_images(file_dir)
# print(sets['train'])

train_gen = DataGenerator(sets['train'], batch_size=batch_size, n_classes=n_classes, augmentation=True)
valid_gen = DataGenerator(sets['valid'], batch_size=batch_size, n_classes=n_classes, augmentation=True)

print(train_gen.__len__())
# train the vox2vox model
h = fit(train_gen, valid_gen, alpha, n_epochs)
