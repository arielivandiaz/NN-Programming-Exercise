import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend 
backend.set_image_dim_ordering('th')


from keras import backend

import argparse
import time

#/********************************************************************************************* 
class model_params():

	neural_network = 0
	optimizator = 0
	activation_1 = 0
	activation_2 = 0
	activation_3 = 0
	activation_4 = 0
	activation_5 = 0

	def __init__ (self):

		args=get_args()

		if (args['Neural Network Selector']):
			self.neural_network =  (args['Neural Network Selector'])
		else :
			self.neural_network = 'Simple_NN'

		if (args['Optimizator']):
			self.optimizator =  (args['Optimizator'])
		else :
			self.optimizator = 'adam'

		if (args['Activation Function Layer 1']):
			self.activation_1 =  (args['Activation Function Layer 1'])
		else:
			self.activation_1= 'relu'

		if (args['Activation Function Layer 2']):
			self.activation_2 =  (args['Activation Function Layer 2'])
		else:
			if (self.neural_network == 'Simple_NN'):
				self.activation_2= 'softmax'
			else :
				self.activation_2= 'relu'

		if (args['Activation Function Layer 3']):
			self.activation_3 =  (args['Activation Function Layer 3'])
		else:
			self.activation_3 = 'relu'

		if (args['Activation Function Layer 4']):
			self.activation_4 =  (args['Activation Function Layer 4'])
		else:
			self.activation_4= 'relu'

		if (args['Activation Function Layer 5']):
			self.activation_4 =  (args['Activation Function Layer 5'])
		else:
			self.activation_4= 'softmax'
	


#/*********************************************************************************************    
def get_args():

	
	parser = argparse.ArgumentParser(description='OSM  UTN FRBB')

	
	parser.add_argument('Neural Network Selector', default=0, nargs='?' )
	parser.add_argument('Optimizator', default='adam', nargs='?',  )
	parser.add_argument('Activation Function Layer 1', default='relu', nargs='?')
	parser.add_argument('Activation Function Layer 2', default='relu', nargs='?')
	parser.add_argument('Activation Function Layer 3', default='relu', nargs='?')
	parser.add_argument('Activation Function Layer 4', default='relu', nargs='?')
	parser.add_argument('Activation Function Layer 5', default='softmax', nargs='?')


	return vars(parser.parse_args())
	


# define baseline model
#/********************************************************************************************* 
def simple_NN(params):
	
	# create model
	model = Sequential()
	#input_function=
	#First Layer
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation=params.activation_1))
	#Last Layer
	model.add(Dense(num_classes, kernel_initializer='normal', activation=params.activation_2))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=params.optimizator, metrics=['accuracy'])
	return model

def simple_CNN(params):
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation=params.activation_1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation=params.activation_2))
	model.add(Dense(num_classes, activation=params.activation_3))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=params.optimizator, metrics=['accuracy'])
	return model

# define the larger model
def large_CNN(params):
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation=params.activation_1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation=params.activation_2))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation=params.activation_3))
	model.add(Dense(50, activation=params.activation_4))
	model.add(Dense(num_classes, activation=params.activation_5))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=params.optimizator, metrics=['accuracy'])
	return model

#/********************************************************************************************* 
def get_data ():
	# load data
	(X_train, y_train), (X_test, y_test) = mnist.load_data()


	# flatten 28*28 images to a 784 vector for each image
	num_pixels = X_train.shape[1] * X_train.shape[2]
	X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
	X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

	# normalize inputs from 0-255 to 0-1
	X_train = X_train / 255
	X_test = X_test / 255


	# normalize inputs from 0-255 to 0-1
	X_train = X_train / 255
	X_test = X_test / 255

	# one hot encode outputs
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	num_classes = y_test.shape[1]

	return 	X_train, y_train, X_test, y_test, num_pixels, num_classes


#/********************************************************************************************* 
def run_evaluation(model, X_train, y_train, X_test, y_test,params):


	file = open('output.txt', 'a')

	write_label(file,params)

	start = time.time()

	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)
	end = time.time()
	print ("Training Time : " , end - start)
	file.write ("Training Time : " + str(end - start))
	file.write ("\t\n")

	start = time.time()
	scores = model.evaluate(X_test, y_test, verbose=1)
	end = time.time()
	print ("Evaluation Time : " , end - start)
	file.write ("Evaluation Time : " + str(end - start))
	file.write ("\t\n")
	

	print("Baseline Error: %.2f%%" % (100-scores[1]*100))

	file.write("Baseline Error: %.2f%%" % (100-scores[1]*100))
	file.write ("\t\n")

	file.close()
	
#/********************************************************************************************* 
def write_label(file,params):

	
	file.write("Test: " + params.neural_network + " \n"))
	file.write("Optimization: " + params.optimizator + " \n"))

	file.write("L1 : " + params.activation_1 + " \n")
	file.write("L2 : " + params.activation_2 + " \n")

	if (params.neural_network=='simple_CNN'):
		file.write("L3 : " + params.activation_3 +" \n")

	elif (params.neural_network=='large_CNN'):
		file.write("L3 : " + params.activation_3 +" \n")
		file.write("L4 : " + params.activation_4 +" \n")
		file.write("L5 : " + params.activation_5 +" \n")





if __name__ == '__main__':


	# fix random seed for reproducibility	
	seed = 7
	numpy.random.seed(seed)

	#Get parameters of the model from arguments
	params = model_params()

	

	#Get model data from minst
	X_train, y_train, X_test, y_test, num_pixels , num_classes= get_data()

	
	# Build the model
	if(params.neural_network=='simple_NN'):
		model = simple_NN(params)
		
	elif (params.neural_network=='simple_CNN'):
		model = simple_CNN(params)

	elif (params.neural_network=='large_CNN'):
		model = large_CNN(params)

	# Fit the model, run evaluation and get output file
	run_evaluation(model, X_train, y_train, X_test, y_test,params)

	# Clean up
	backend.clear_session()

