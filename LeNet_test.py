from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import csv
from os import listdir
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from imutils import paths

norm_size = 32
labelPath = "./cs-ioc5008-hw1/dataset/dataset/train"
basePath = "./cs-ioc5008-hw1/dataset/dataset/test"

def predict(path):
	print("[INFO] loading network...")
	model = load_model("best_model.h5")
	labels = listdir(labelPath)
	#print(labels)
	imagePaths = sorted(list(paths.list_images(path)))
	with open('result.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['id', 'label'])
		for imagePath in imagePaths:
			imageID = imagePath[imagePath.rfind("\\") + 1:]
			imageID = imageID.replace(".jpg", "")
			image = cv2.imread(imagePath)
			image = cv2.resize(image, (norm_size, norm_size))
			image = image.astype("float") / 255.0
			image = img_to_array(image)
			image = np.expand_dims(image, axis=0)
			result = model.predict(image)[0]
			proba = np.max(result)
			label = int(np.where(result==proba)[0])
			writer = csv.writer(csvfile)
			writer.writerow([imageID, labels[label]])
	
if __name__ == '__main__':
	predict(basePath)