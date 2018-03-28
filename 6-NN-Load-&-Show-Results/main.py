import scipy.misc
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import os



def load_params ():


	file = open('params.txt', 'r') 

	optimizator=file.readline()
	loss_function=file.readline()
	b_size = file.readline()

	file.close()

	return optimizator[:-1],loss_function[:-1],b_size[:-1]

#/********************************************************************************************* 
def get_max(vector):
	the_max=-1
	i_max=-1

	for i in range(len(vector)):
		if the_max<vector[i]:
			i_max=i
			the_max=vector[i]
	return i_max

#/********************************************************************************************* 
def create_directories ():

		folder= 'Results as Images'
		if os.path.exists(folder) == 0:
			print ('Folder Created :')
			os.mkdir(folder)
		for i in range(0,10):
			if os.path.exists(folder+'\\'+str(i)) == 0:
				print ('Folder Created :')
				os.mkdir(folder+'\\'+str(i))

#/********************************************************************************************* 
def save_images(prediction_prob,prediction,X_image):

	folder= 'Results as Images\\'
	for i in range(len(prediction)):

		print('Img '+str(i)+' is  = '+str(prediction[i])+' with prob : ' + str(prediction_prob[i][prediction[i]]))
		image_name=folder+str(prediction[i])+'\\'+'img'+str(i)+'_'+str(prediction_prob[i][prediction[i]]*100)[:3].replace('.', 'P')+'.jpg'
		
		scipy.misc.imsave(image_name, X_image[i])

#/********************************************************************************************* 
def get_data ():
	# load data
	(X_test, y_train), (X_test, y_test) = mnist.load_data()

	X_image= X_test

	num_pixels = X_test.shape[1] * X_test.shape[2]
	X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

	# normalize inputs from 0-255 to 0-1
	X_test = X_test / 255

	# normalize inputs from 0-255 to 0-1
	X_test = X_test / 255

	# one hot encode outputs
	y_test = np_utils.to_categorical(y_test)


	return 	X_test, y_test, X_image


#/********************************************************************************************* 
if __name__ == '__main__':

	
	optimizator, loss_function ,b_size= load_params()

	# load json and create model
	X_test, y_test, X_image = get_data()

	
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Model loaded")
	 
	# evaluate loaded model on test data
	loaded_model.compile(loss=loss_function, optimizer=optimizator, metrics=['accuracy'])
	
	# use the model for predicts outputs
	prediction_prob = loaded_model.predict(X_test, batch_size=int(b_size))

	# get the digit with best probability
	prediction = []
	for i in prediction_prob:
		prediction.append(get_max(i))
	
	create_directories()

	save_images(prediction_prob,prediction,X_image)
	
	

