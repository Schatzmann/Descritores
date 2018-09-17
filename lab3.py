#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Autor:
#Annelyse Schatzmann         GRR20151731


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import descritores as d

from sklearn.datasets import load_svmlight_file

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import cPickle

#_______________ KNN _______________#

def knn(x_data, y_data , nome):
	
	k_range = range(1, 16)
	k_scores = []

	for k in k_range:
	  knn = KNeighborsClassifier(n_neighbors=k)
	  scores = cross_val_score(knn, x_data, y_data, cv=5, scoring='accuracy')
	  k_scores.append(scores.mean())


	plt.plot(k_range, k_scores, linestyle='-', color='blue', marker="*", label="Val. Cruzada")
	plt.legend(loc='best')
	plt.grid(True)
	plt.xlabel('Valores de K')
	plt.ylabel('Acuracia')
	plt.title(nome)
	 
	plt.show()


#_______________ DIGITS DATASET _______________#

def digits():
	print ('Carregando imagens...')
	path_images = 'digits/data'
	archives = os.listdir(path_images)

	images = []
	labels = []
	arq = open('digits/files.txt')
	lines = arq.readlines()
	print ('Extraindo as features')

	for line in lines:
		aux = line.split('/')[1]
		image_name = aux.split(' ')[0]
		label = line.split(' ')[1]
		label = label.split('\n')[0]
			
		for archive in archives:
			if archive == image_name:
				image = cv2.imread(path_images +'/'+ archive, 0)
				d.descrit_digits(image, label)


	x_data, y_data = load_svmlight_file('caracteristicas.txt')
	knn(x_data, y_data, 'Digits Dataset')

	os.system("rm caracteristicas.txt")


#_______________ CIFAR 10_______________#


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """
    data_path = "data/CIFAR-10/"
    _get_file_path = os.path.join(data_path, "cifar-10-batches-py/", filename)
    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images



def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls

def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """
    

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(5):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)




if __name__ == "__main__":
    digits()
  # Width and height of each image.
  # img_size = 32

  # # Number of channels in each image, 3 channels: Red, Green, Blue.
  # num_channels = 3
  # _num_images_train = 5 * 10000
  # load_training_data()
	

	