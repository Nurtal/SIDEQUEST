## Learn model


## Genetic algorithm to play with model structure of CNN


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import tensorflow as tf
from keras import backend as K

import random
import matplotlib.pyplot as plt
from scipy.misc import imread
import glob
import random


import classifier



def generate_random_model(nb_class, ip_shape):
	##
	## IN PROGRESS
	## 
	## TODO : dropout

	## Parameters
	## === FIX ===
	## nb_class -> number of classes
	## ip_shape -> input shape (image dimmensions)

	## === LEARNED ===
	## k_size -> kernel_size
	## nb_hl -> number of hidden layers
	## max_output_size ->  the max dimensionality of the output space


	## init parameters to learn
	k_size = random.randint(2,4)
	k_size = (k_size,k_size)
	nb_hl = random.randint(1,5)
	max_output_size = random.randint(128,256)
	filters = random.randint(32,max_output_size)


	## init model
	fashion_model = Sequential()

	## input layer
	fashion_model.add(Conv2D(filters, kernel_size=k_size,activation='linear',input_shape=ip_shape,padding='same'))
	fashion_model.add(LeakyReLU(alpha=0.1))
	fashion_model.add(MaxPooling2D((2, 2),padding='same'))

	## hidden layers
	for x in xrange(0,nb_hl):

		k_size = random.randint(2,16)
		k_size = (k_size,k_size)
		filters = random.randint(32,max_output_size)

		fashion_model.add(Conv2D(filters, k_size, activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))                  
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

	## output stuff
	fashion_model.add(Flatten())	
	fashion_model.add(Dense(max_output_size, activation='linear'))
	fashion_model.add(LeakyReLU(alpha=0.1))                  
	fashion_model.add(Dense(nb_class, activation='softmax'))

	return fashion_model



def stupid_optimization(max_number_of_layers):
	##
	## IN PROGRESS
	##
	## Exaustif search
	## on number of layers
	##

	## init parameters to learn
	k_size = random.randint(2,16)
	max_output_size = random.randint(128,1084)
	filters = random.randint(32,max_output_size)

	for x in xrange(1,max_number_of_layers):

		choucroute = 1
		## input layer


		## output layer




def cross_model(male, female, mutation_rate):
	##
	## IN PROGRESS
	##
	## reprodutcion function
	##


	
	## determine max lenght
	max_size = "NA"
	max_filter = "NA"

	child_is_ok = True

	if(random.randint(0,100) > 50):
		max_size = len(male.layers)
		matrix_parent = male
		side_parent = female
	else:
		max_size = len(female.layers)
		matrix_parent = female
		side_parent = male


	# get max filter values
	filters_values = []
	for layer in matrix_parent.layers:
		config = layer.get_config()
		try:
			filter_value = config['filters']
		except:
			filter_value = 0
		filters_values.append(int(filter_value))
	max_filter = max(filters_values)


	## Create child
	child = Sequential()
	
	## add input layer
	child.add(matrix_parent.layers[0])
	child.add(matrix_parent.layers[1])
	child.add(matrix_parent.layers[2])

	## Hidden layers
	for x in xrange(0,len(matrix_parent.layers[3:-4]) / 3):

		## legacy of filters
		filters = "NA"
		if(random.randint(0,100) > 40):
			config = matrix_parent.layers[3+(x*3)].get_config()
			filters = config['filters']
		else:
			try:
				config = side_parent.layers[3+(x*3)].get_config()
				filters = config['filters']
			except:
				config = matrix_parent.layers[3+(x*3)].get_config()
				filters = config['filters']

			if(filters > max_filter):
				filters = max_filter

		## legacy of kernel size
		k_size = "NA"
		if(random.randint(0,100 > 40)):
			config = matrix_parent.layers[3+(x*3)].get_config()
			k_size = config['kernel_size']
		else:
			try:
				config = side_parent.layers[3+(x*3)].get_config()
				k_size = config['kernel_size']
			except:
				config = matrix_parent.layers[3+(x*3)].get_config()
				k_size = config['kernel_size']

		## mutation
		if(random.randint(0,100) < mutation_rate):
			k_size = random.randint(1,16)
			k_size = (k_size,k_size)
			filters = random.randint(32,max_filter)

		## add layer to child
		child.add(Conv2D(filters, k_size, activation='linear',padding='same'))
		child.add(LeakyReLU(alpha=0.1))                  
		child.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

	
	## add output layer
	## can produce unfunctionnal network
	try:
		for layer in matrix_parent.layers[-4:]:
			child.add(layer)
	except:
		child_is_ok = False


	## return child
	if(child_is_ok):
		return child
	else:
		return 0





def compute_model_score(model,train_X,train_Y,test_X,test_Y, id_run):
	##
	## IN PROGRESS
	##

	"""
	config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1, 
                            allow_soft_placement=True)    
    session = tf.Session(config=config)
    K.set_session(session)
	"""

	## parameters
	batch_size = 32
	epochs = 2
	num_classes = 2

	## data pre-processing
	train_X = train_X.astype('float32')
	test_X = test_X.astype('float32')
	train_X = train_X / 255.
	test_X = test_X / 255.

	# Change the labels from categorical to one-hot encoding
	train_Y_one_hot = to_categorical(train_Y)
	test_Y_one_hot = to_categorical(test_Y)

	## split dtaa
	train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

	## prepare the model
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	print model.summary()

	## train the model
	fashion_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

	# evaluate the model
	log_file = open("log/"+str(id_run)+".log", "w")
	test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
	log_file.write('Test loss:'+str(test_eval[0]))
	log_file.write('Test accuracy:'+str(test_eval[1]))
	log_file.close()

	## create a few figures
	accuracy = fashion_train.history['acc']
	val_accuracy = fashion_train.history['val_acc']
	loss = fashion_train.history['loss']
	val_loss = fashion_train.history['val_loss']
	epochs = range(len(accuracy))
	plt.figure()
	plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
	plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig("log/"+str(id_run)+"_acc.png")
	plt.close()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig("log/"+str(id_run)+"_loss.png")
	plt.close()

	## return accuracy as a score
	return test_eval[1]



def population_evaluation(population, good_parents, bad_parents):
	##
	## IN PROGRESS
	##

		
	## Compute the score for each individual (matrix) in the
	## population.
	model_index_to_score = {}
	index_to_model = {}
	model_index = 0
	channel = "all"
	(X_train, Y_train), (X_validation, Y_validation) = classifier.load_images(channel)

	for model in population:
		model_index_to_score[model_index] = compute_model_score(model, X_train, Y_train, X_validation, Y_validation, "model_"+str(model_index))
		index_to_model[model_index] = model
		model_index += 1


	## compute population score
	scores = model_index_to_score.values()
	pop_score = mean(scores)


	## Select good parents, i.e the top score
	all_good_parents_assigned = False
	number_of_good_parents = 0
	list_of_good_parents = []
	while(not all_good_parents_assigned):

		selected_parent = max(model_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
		del model_index_to_score[selected_parent]
		list_of_good_parents.append(selected_parent)
		number_of_good_parents += 1

		if(number_of_good_parents == good_parents):
			all_good_parents_assigned = True

	## Select bad parents, i.e the low score
	all_bad_parents_assigned = False
	number_of_bad_parents = 0
	list_of_bad_parents = []
	while(not all_bad_parents_assigned):

		selected_parent = min(model_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
		del model_index_to_score[selected_parent]
		list_of_bad_parents.append(selected_parent)
		number_of_bad_parents += 1

		if(number_of_bad_parents == bad_parents):
			all_bad_parents_assigned = True

	## Create the list of patient to return
	parents_id = list_of_good_parents + list_of_bad_parents
	parents = []
	for p_id in parents_id:
		parents.append(index_to_model[p_id])

	return (parents, pop_score)








def evolve():

	##
	## IN PROGRESS
	##

	population_size = 25
	ip_shape = (480,480,3)
	nb_class = 2
	nb_cyles = 100
	mutation_rate = 10

	log_file = open("evolution.log", "w")


	## generate initial pop
	population = []
	for x in xrange(0,population_size):
		model = generate_random_model(nb_class, ip_shape)
		population.append(model)

	for c in xrange(0, nb_cyles):
		
		new_population = []

		## select parents
		parents, pop_score = population_evaluation(population,10,4)

		## write pop score
		log_file.write("Generation "+str(c)+";score "+str(pop_score)+"\n")

		## reproduce
		individual_cmpt = 0
		while(individual_cmpt != population_size):

			## get the parents (random selection)
			parents_are_different = False
			male_index = -1
			female_index = -1
			while(not parents_are_different):
				
				male_index = random.randint(0,len(parents))
				female_index = random.randint(0,len(parents))

				if(male_index != female_index):
					parents_are_different = True

			parent_male = parents[random.randint(0,len(parents)-1)]
			parent_female = parents[random.randint(0,len(parents)-1)]

			## create the child
			child = cross_model(parent_male, parent_female, mutation_rate)

			if(child):
				new_population.append(child)
				individual_cmpt += 1

		## next cycle
		population = new_population

	log_file.close()







evolve()

#stuff = (480,480,3)

#truc = generate_random_model(2, stuff)
#machin = generate_random_model(2, stuff)

#cross_model(truc, machin, 10)

#channel = "PC5.5.A_APC.AF750.A"
#(X_train, Y_train), (X_validation, Y_validation) = classifier.load_images(channel)

"""
import multiprocessing
ls = [1,2,3]
pool = multiprocessing.Pool()
results = pool.map(train, ls)
pool.close()
pool.terminate()
"""
#compute_model_score(truc, X_train, Y_train, X_validation, Y_validation, "truc_model_on_test_channel")


#truc.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#truc.summary()





