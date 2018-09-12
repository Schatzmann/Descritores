#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Autor:
#Annelyse Schatzmann     GRR20151731


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import cPickle
from skimage.feature import local_binary_pattern



img = cv2.imread('teste.tif',0)
ret, thresh = cv2.threshold (img, 127, 255, 0) 
img2, contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
#Cada contorno é armazenado como um vetor de pontos.

carac_img = np.array([])
caracteristicas = np.array([])

#_______________ HU _______________#

for i in range(len(contornos)):
	cnt = contornos[i]	
	moments = cv2.moments(cnt)
hu = cv2.HuMoments(moments)

# cx = int(moments['m10']/moments['m00'])
# cy = int(moments['m01']/moments['m00'])
# centroides


#_______________ LBP _______________#

radius = 3
n_points = 8 * radius

lbp = local_binary_pattern(img, n_points, radius, method='uniform')
n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp, normed = True, bins = n_bins, range =(0, n_bins))
#Calcule o histograma de um conjunto de dados.
# O histograma do resultado da LBP é uma boa medida para classificar as texturas.


#_______________ Convex Hull _______________#

area = cv2.contourArea(cnt)
#Calcula uma área de contorno.(número de pixels diferentes de zero)
hull = cv2.convexHull(cnt)
#Encontra o casco convexo de um conjunto de pontos.
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
#Solidez é a relação entre a área de contorno e a área convexa do casco.
#solidity = float(moments['m00'])/hull_area




carac_img = np.concatenate((carac_img,hu), axis = None)
carac_img = np.concatenate((carac_img,hist), axis = None)
carac_img = np.concatenate((carac_img,solidity), axis = None)


print(hu)
print('\n')
print(hist)
print('\n')
print(solidity)
print('\n')
print(carac_img)
#_____________________________________img2______________________________________#

