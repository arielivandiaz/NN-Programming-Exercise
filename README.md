# Welcome to Neural Network World!

In this repo it is proposed to solve the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) problem of the identification of handwritten digits through neural networks.


# Folders

**1-ImportData:** is an example of how to import the MNIST dataset.

**2-SimpleNN:**  is a first example and very good approach to solve the problem using a single layer neural network.

**3-SimpleCNN:** is a next example of how to use a convolutional neural network to solve the problem.

**4-LargeCNN:** is an example of a Large CNN, a multi-layered convolutional neural network
**5-NN-Research:** in this folder there is a complete program that allows to execute one of the 3 neural networks with the custom parameters. There is also the Test function package that was used to prepare the report. The final neural network is saved in a JSON and H5 format in order to be reused and evaluated by the content of the next folder.

**6-NN-Load-&-Show-Results:** This folder contains a program that allows you to take a neural network stored in the format of the folder 5-NN-Research, run a prediction with this and store the outputs as image files, in the folder corresponding to the output obtained from the network.
The image stored in folder "1" means that the network predicted that this image corresponds to digit one. The format of the name of the image is for example "img5_96P" where img5 specifies the number of the image and 96P that the probability was 96%.


## To run

**1-ImportData:**  python main.py

**2-SimpleNN:** python main.py

**3-SimpleCNN:** python main.py

**4-LargeCNN:** python main.py

**5-NN-Research:**
  python main.py simple_NN nadam binary_crossentropy 100 15 relu softmax


> Parameter n°1 (Simple_NN) : is the choice of which network you want to use.

> Parameter n°2 ( nadam ) : is the [optimization](https://keras.io/optimizers/) function 

> Parameter n°3 (binary_crossentropy ) : the [loss function](https://keras.io/losses/).

> Parameter n°4 (100 ) : trains the model for a fixed number of epochs (iterations on a dataset).

> Parameter n°5 (relu) : the [activation function](https://keras.io/activations/) of the layer n°1, hidden layer in this case.

> Parameter n°6 (softmax) : activation of layer 2, output layer in this case, softmax transforms the output into a probability.

> Parameter n°7: activation of layer 3, the last layer of the SimpleCNN.

> Parameter n°8: activation function of layer 4, only for LargeCNN.

> Parameter n°9: activation of layer 5, output layer of LargeCNN.


All the parameters are optional, by default you have a set of preset parameters.

If you want to choose a CUDAdevice available (second device in this case, device 1):

CUDA_VISIBLE_DEVICES=1  python main.py 

To run only on CPU without CUDAdevice :

CUDA_VISIBLE_DEVICES=""  python main.py 

**6-NN-Load-&-Show-Results:** python main.py

You must empty or delete the "Results as Images" directory before collecting new data.

## Best Result
At the moment, it has only been possible to evaluate SimpleNN in depth with very good results:

**Best Performance**
>python main.py simple\_NN nadam binary\_crossentropy 15 15 relu softmax

Training Time :  1073.427568912506

10000/10000 - 1s 74us/step

Evaluation Time :  0.7415130138397217

Baseline Error: 0.42%%

loss: 2.33%

acc: 99.58%

**Best Rate**
>python main.py simple_NN nadam binary_crossentropy 100 15 relu softmax

Training Time : ', 7158.396837949753

Evaluation Time :  0.7365529537200928

10000/10000 - 1s 74us/step

Baseline Error: 0.41%

loss: 2.28%

acc: 99.59%


## About this work

This work is a TP project of the class  "Intelligent Systems: Reasoning and Recognition", ENSIMAG, Grenoble INP, Grenoble, France.
May 2018.
