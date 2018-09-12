#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Autor:
#Annelyse Schatzmann         GRR20151731


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import cPickle
from skimage.feature import local_binary_pattern



#_______________ KNN _______________#

def knn(caracteristicas):
	
	k_range = range(1, 31)
	k_scores = []

	for k in k_range:
	  knn = KNeighborsClassifier(n_neighbors=k)
	  # knn.fit(X,y)
	  scores = cross_val_score(knn, caracteristicas, caracteristicas, cv=5, scoring='accuracy')
	  k_scores.append(scores.mean())


	plt.plot(k_range, k_scores, linestyle='-', color='blue', marker="*", label="Val. Cruzada")
	plt.legend(loc='best')
	plt.grid(True)
	plt.xlabel('Valores de K')
	plt.ylabel('Acuracia')
	plt.title("Digits Dataset")
	 
	plt.show()




#------------- DIGITS DATASET-------------#

print ('Carregando imagens...')
path_images = 'digits/data'
archives = os.listdir(path_images)

images = []
arq = open('digits/files.txt')
lines = arq.readlines()
print ('Extracting dummy features')

for line in lines:
	aux = line.split('/')[1]
	image_name = aux.split(' ')[0]
	label = line.split(' ')[1]
	label = label.split('\n')
		
	for archive in archives:
		if archive == image_name:
			# image = cv2.imread(path_images, 0)
			image = cv2.imread(path_images +'/'+ archive, 0)
			# img = cv2.imread('teste.tif')
			ret, thresh = cv2.threshold (image, 127, 255, 0) 
			img2, contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
			


#------------- CIFAR-10 -------------#
