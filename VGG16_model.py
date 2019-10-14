from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import InputLayer
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as k
from keras.applications import VGG16

weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		model_vgg = VGG16(weights=weights_path, include_top=False, input_shape=(height, width, depth))
		for layer in model_vgg.layers[:-9]:
			layer.trainable = False
		net = Sequential()
		net.add(model_vgg)
		net.add(GlobalAveragePooling2D())
		net.add(Dropout(0.2))
		net.add(Dense(100, activation='relu'))
		net.add(Dense(classes, activation='softmax'))
		return net