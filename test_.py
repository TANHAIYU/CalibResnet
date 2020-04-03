import glob
import os, re, sys
import pickle
from typing import Optional, Any
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset
import os,sys,argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from dataset import trainset,testset

IMAGE_FILE_PATH_DISTORTED = "/home/haiyutan/master-thesis/images/dataset/inceptionv3_test_discrete_add_dc_distortionCenter/"

focal_start = 503.2
focal_end = 934.51
dist_end = 0.8
IMAGE_WIDTH_ORIG = 1241

paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'train/' + "*.png")
paths_test = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'test/' + "*.png")
train_parameters = []
test_parameters = []

for path in paths_train:
	curr_parameter = float(re.split('_', path)[9])
	curr_parameter*718.8562/float(IMAGE_WIDTH_ORIG)
	labels_focal_train = float((curr_parameter*718.8562/float(IMAGE_WIDTH_ORIG))-0.34755336)/(1.5-0.34755336)  # normalize bewteen 0 and 1
	curr_parameter = float(re.split('_', path)[11])
	labels_dc_x_train = float((curr_parameter-0.45)/0.1)  # normalize bewteen 0 and 1
	curr_parameter = float(re.split('.png', re.split('_', path)[12])[0])
	labels_dc_y_train = float((curr_parameter-0.45)/0.1)  # normalize bewteen 0 and 1
	curr_parameter = float(re.split('_', path)[7])
	labels_distortion_train = float((curr_parameter+1)/2.0)
	train_parameters.append([labels_dc_x_train, labels_dc_y_train, labels_focal_train, labels_distortion_train])

for path in paths_test:
	curr_parameter = float(re.split('_', path)[9])
	labels_focal_test = float((curr_parameter*718.8562/float(IMAGE_WIDTH_ORIG))-0.34755336)/(1.5-0.34755336)  # normalize bewteen 0 and 1
	curr_parameter = float(re.split('_', path)[11])
	labels_dc_x_test =float((curr_parameter-0.45)/0.1)  # normalize bewteen 0 and 1
	curr_parameter = float(re.split('.png', re.split('_', path)[12])[0])
	labels_dc_y_test = float((curr_parameter-0.45)/0.1)  # normalize bewteen 0 and 1
	curr_parameter = float(re.split('_', path)[7])
	labels_distortion_test = float((curr_parameter+1)/2.0)
	test_parameters.append([labels_dc_x_test, labels_dc_y_test, labels_focal_test, labels_distortion_test])


# print(paths_train)
train_data  = trainset()
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
# img,labels,len = DataLoader.__iter__().__next__()
