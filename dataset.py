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
np.save("train_parameters.npy", train_parameters)
np.save("test_parameters.npy", test_parameters)

paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'train/' + "*.png")
file_train = paths_train
file_test = paths_test
print(file_train)
print(file_test)

np.save("file_train.npy", file_train)
np.save("file_test.npy", file_test)  # 里面是图片的路径

preprocess = transforms.Compose([
	# transforms.Scale(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()
])


def default_loader(path):
	img_pil = Image.open(path).convert('RGB')
	# img_pil = img_pil.resize((224,224))
	img_tensor = preprocess(img_pil)
	return img_tensor


class trainset(Dataset):
	def __init__(self, loader=default_loader):
		# 定义好 image 的路径
		self.images = file_train
		self.target = train_parameters
		self.loader = loader

	def __getitem__(self, index):
		fn = self.images[index]
		images = self.loader(fn)
		labels = self.target[index]
		labels = torch.FloatTensor(labels)
		return images, labels

	def __len__(self):
		return len(self.images)


class testset(Dataset):
	def __init__(self, loader=default_loader):
		# 定义好 image 的路径
		self.images = file_test
		self.target = test_parameters
		self.loader = loader

	def __getitem__(self, index):
		fn = self.images[index]
		images = self.loader(fn)
		labels = self.target[index]
		return images, labels

	def __len__(self):
		return len(self.images)

# class DataTrain(Dataset):
#
# 	def __init__(self, path, transform=None):
# 		# if transform is given, we transoform data using
# 		with open(os.path.join(path, 'train'), 'rb') as cifar100:
# 			self.data = pickle.load(cifar100, encoding='bytes')
# 		self.transform = transform
#
# 	def __len__(self):
# 		return len(self.data['fine_labels'.encode()])
#
# 	def __getitem__(self, index):
# 		label = self.data['fine_labels'.encode()][index]
# 		r = self.data['data'.encode()][index, :1024].reshape(32, 32)
# 		g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
# 		b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
# 		image = np.dstack((r, g, b))
#
# 		if self.transform:
# 			image = self.transform(image)
# 		return label, image

# class CIFAR100Test(Dataset):
#
# 	def __init__(self, path, transform=None):
# 		with open(os.path.join(path, 'test'), 'rb') as cifar100:
# 			self.data = pickle.load(cifar100, encoding='bytes')
# 		self.transform = transform
#
# 	def __len__(self):
# 		return len(self.data['data'.encode()])
#
# 	def __getitem__(self, index):
# 		label = self.data['fine_labels'.encode()][index]
# 		r = self.data['data'.encode()][index, :1024].reshape(32, 32)
# 		g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
# 		b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
# 		image = np.dstack((r, g, b))
#
# 		if self.transform:
# 			image = self.transform(image)
# 		return label, image
