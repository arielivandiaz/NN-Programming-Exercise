# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_test[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_test[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()