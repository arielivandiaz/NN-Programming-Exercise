# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
import random
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot some images as gray scale

plt.subplot(331)
plt.axis('off')
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.axis('off')
plt.imshow(X_test[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.axis('off')
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(334)
plt.axis('off')
plt.imshow(X_test[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.axis('off')
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.axis('off')
plt.imshow(X_test[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(337)
plt.axis('off')
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(338)
plt.axis('off')
plt.imshow(X_test[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))
plt.subplot(339)
plt.axis('off')
plt.imshow(X_train[random.randint(0, len(X_test))], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()