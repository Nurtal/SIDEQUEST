import matplotlib
matplotlib.use('TkAgg')
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.misc import imread
import glob
import random


def load_images():
	##
	## IN PROGRESS
	##
	## load data for CNN, from
	## images to matrix.
	##
	## kind of alpha version, just pick one combination
	## of channels.
	##
	## y label:
	##     0 -> uncompensated data
	##     1 -> compensated data
	##

	## proportion of train and validation
	train_proportion = 0.7

	## Select a single comination of channel to start
	channel_selected = "FITC.A_APC.AF750.A"
	raw_files = glob.glob("data/images/raw/*"+str(channel_selected)+".png")
	compensation_files = glob.glob("data/images/compensated/*"+str(channel_selected)+".png")

	## create train dataset
	X_train = []
	Y_train = []
	X_validation = []
	Y_validation = []
	
	image_in_train_file = []
	number_of_train_cases = int(float(len(raw_files))*float(train_proportion))
	number_of_validation_cases = len(raw_files) - number_of_train_cases

	## raw
	image_keep = 0
	while(image_keep < number_of_train_cases):
		for image in raw_files:

			## roll the dice
			if(random.randint(0,100) > 50):
				if(image not in image_in_train_file):
					matrix = imread(image)
					X_train.append(matrix)
					Y_train.append(0)
					image_in_train_file.append(image)
					image_keep += 1


	## compensated
	number_of_train_cases = int(float(len(compensation_files))*float(train_proportion))
	image_keep = 0
	while(image_keep < number_of_train_cases):
		for image in compensation_files:

			## roll the dice
			if(random.randint(0,100) > 50):
				if(image not in image_in_train_file):
					matrix = imread(image)
					X_train.append(matrix)
					Y_train.append(1)
					image_in_train_file.append(image)
					image_keep += 1



	## Create validation dataset
	for image in raw_files:
		if(image not in image_in_train_file):
			matrix = imread(image)
			X_validation.append(matrix)
			Y_validation.append(0)

	for image in compensation_files:
		if(image not in image_in_train_file):
			matrix = imread(image)
			X_validation.append(matrix)
			Y_validation.append(1)

	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)
	X_validation = np.asarray(X_validation)
	Y_validation = np.asarray(Y_validation)

	return (X_train, Y_train), (X_validation, Y_validation)



def run_CNN():
	##
	## Very immature step
	## Just run the CNN for now,
	## basic structure, no smart transformation ...
	##
	## WORK IN PROGRESS
	##

	(train_X,train_Y), (test_X,test_Y) = load_images()

	print('Training data shape : ', train_X.shape, train_Y.shape)
	classes = np.unique(train_Y)
	nClasses = len(classes)
	print('Total number of outputs : ', nClasses)
	print('Output classes : ', classes)
	plt.figure(figsize=[5,5])

	# Display the first image in training data
	plt.subplot(121)
	plt.imshow(train_X[0,:,:], cmap='gray')
	plt.title("Ground Truth : {}".format(train_Y[0]))

	# Display the first image in testing data
	plt.subplot(122)
	plt.imshow(test_X[0,:,:], cmap='gray')
	plt.title("Ground Truth : {}".format(test_Y[0]))
	#plt.show()
	plt.close()


	train_X = train_X.astype('float32')
	test_X = test_X.astype('float32')
	train_X = train_X / 255.
	test_X = test_X / 255.

	# Change the labels from categorical to one-hot encoding
	train_Y_one_hot = to_categorical(train_Y)
	test_Y_one_hot = to_categorical(test_Y)

	# Display the change for category label using one-hot encoding
	print('Original label:', train_Y[0])
	print('After conversion to one-hot:', train_Y_one_hot[0])

	from sklearn.model_selection import train_test_split
	train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

	train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


	import keras
	from keras.models import Sequential,Input,Model
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.advanced_activations import LeakyReLU

	batch_size = 32
	epochs = 15
	num_classes = 2

	fashion_model = Sequential()
	fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(480,480,3),padding='same'))
	fashion_model.add(LeakyReLU(alpha=0.1))
	fashion_model.add(MaxPooling2D((2, 2),padding='same'))
	fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	fashion_model.add(LeakyReLU(alpha=0.1))
	fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	fashion_model.add(LeakyReLU(alpha=0.1))                  
	fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	fashion_model.add(Flatten())
	fashion_model.add(Dense(128, activation='linear'))
	fashion_model.add(LeakyReLU(alpha=0.1))                  
	fashion_model.add(Dense(num_classes, activation='softmax'))

	fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	fashion_model.summary()

	fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

	test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
	print('Test loss:', test_eval[0])
	print('Test accuracy:', test_eval[1])


	accuracy = fashion_train.history['acc']
	val_accuracy = fashion_train.history['val_acc']
	loss = fashion_train.history['loss']
	val_loss = fashion_train.history['val_loss']
	epochs = range(len(accuracy))
	plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
	plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()
	plt.close()
