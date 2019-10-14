from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization
from keras.layers import InputLayer
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as k

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		if k.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
		#first conv, BatchNormalization, relu, maxPooling
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
		#second conv, BatchNormalization, relu, maxPooling
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
		#first fully connected layer
		model.add(Flatten())
		model.add(Dense(500))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		#second fully connected layer
		model.add(Dense(130))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
		#softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model