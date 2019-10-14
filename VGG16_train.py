from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
from keras.callbacks import ModelCheckpoint
from keras.applications import imagenet_utils
import numpy as np
import random
import cv2
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from os import listdir
from os.path import join
import sys
sys.path.append('..')
from VGG16_model import VGGNet

EPOCHS = 15
INIT_LR = 0.00001
BS = 20
CLASS_NUM = 13
norm_size = 224
basePath = "./cs-ioc5008-hw1/dataset/dataset/train"

def load_data(path):
	print("[INFO] loading images...")
	data = []
	imagePaths = []
	labels = listdir(path)
	num_labels = []
	num = 0
	for l in labels:
		#path of every label
		labelPath = join(path, l)
		#path of every image
		imagePaths = imagePaths + sorted(list(paths.list_images(labelPath)))
		#label of every image
		num_labels = num_labels + [num]*len(list(paths.list_images(labelPath)))
		num = num + 1
	tmp_list = list(zip(imagePaths, num_labels))
	random.seed(42)
	#shuffle the images and labels
	random.shuffle(tmp_list)
	imagePaths, num_labels = zip(*tmp_list)
	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (norm_size, norm_size))
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)
		data.append(image)
	trainX = data[int(len(data)/10):]
	valX = data[:int(len(data)/10)]
	trainX = np.vstack(trainX)
	valX = np.vstack(valX)
	num_labels = np.array(num_labels)
	num_labels = to_categorical(num_labels, num_classes=CLASS_NUM)
	trainY = num_labels[int(len(data)/10):]
	valY = num_labels[:int(len(data)/10)]
	return trainX, trainY, valX, valY

def train(trainX, trainY, valX, valY):
	print("[INFO] compiling model...")
	model = VGGNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
	opt = Adam(lr=INIT_LR)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	
	print("[INFO] training network...")
	checkpoint = ModelCheckpoint("best_VGG16model.h5", verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
	H = model.fit(trainX, trainY, batch_size=BS, validation_data=(valX, valY), epochs=EPOCHS, verbose=1, callbacks=[checkpoint])
	
	print("[INFO] saving network...")
	model.save("VGG16model.h5")
	del model
	
if __name__=='__main__':
	trainX, trainY, valX, valY = load_data(basePath)
	train(trainX, trainY, valX, valY)