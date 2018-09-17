#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Autor:
#Annelyse Schatzmann     GRR20151731


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern



def descrit_digits(image, label):

	ret, thresh = cv2.threshold(image, 127, 255, 0) 

	if cv2.__version__.startswith('3.'):
		img2, contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
		#Cada contorno é armazenado como um vetor de pontos.
	else:
		contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	carac_img = np.array([])
	caracteristicas = np.array([])

	#_______________ HU _______________#

	for i in range(len(contornos)):
		cnt = contornos[i]	
		moments = cv2.moments(cnt)
		hu = cv2.HuMoments(moments)
		carac_img = np.concatenate((carac_img,hu), axis = None)

	# cx = int(moments['m10']/moments['m00'])
	# cy = int(moments['m01']/moments['m00'])
	# centroides
	# print(hu)
	#_______________ HOG _______________#

	samples = []
	for i in range(len(contornos)):
		gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
		#Calcula a primeira, segunda, terceira ou derivada de imagem
		mag, ang = cv2.cartToPolar(gx, gy)
		#usada para calcular a magnitude e o ângulo de vetores 2D
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)

		# eps = 1e-7
		# hist /= hist.sum() + eps
		# hist = np.sqrt(hist)
		# from numpy.linalg import norm
		# hist /= norm(hist) + eps
  #   # transform to Hellinger kernel
    
		samples.append(hist)
	carac_img = np.concatenate((carac_img,samples), axis = None)

	#_______________ Convex Hull _______________#

	for i in range(len(contornos)):
		cnt = contornos[i]
		area = cv2.contourArea(cnt)
		#Calcula uma área de contorno.(número de pixels diferentes de zero)
		hull = cv2.convexHull(cnt)
		#Encontra o casco convexo de um conjunto de pontos.
		hull_area = cv2.contourArea(hull)
		if (hull_area == 0):
			hull_area = 1
		solidity = float(area)/hull_area
		#Solidez é a relação entre a área de contorno e a área convexa do casco.
		#solidity = float(moments['m00'])/hull_area
		carac_img = np.concatenate((carac_img,solidity), axis = None)
	
	

	escrita = '%s' % (label)
	for i in range(1, len(carac_img)):
		str_aux = ' ' + str(i)+ ":%f" % (carac_img[i])
		escrita = escrita + str_aux
	escrita = escrita + '\n'
	# colocar label+caracteristicas em uma string só

	arq = open('caracteristicas.txt','a')
	arq.write(escrita)
	arq.close(	)



def descrit_cif(image, label):

	ret, thresh = cv2.threshold(image, 127, 255, 0) 

	if cv2.__version__.startswith('3.'):
		img2, contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
		#Cada contorno é armazenado como um vetor de pontos.
	else:
		contornos, hierarquia = cv2.findContours (thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	carac_img = np.array([])
	caracteristicas = np.array([])


	#_______________ SIFT _______________#

	# image = cv2.imread("test_image.jpg")
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	(kps, descs) = sift.detectAndCompute(thresh, None)
	print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

	#_______________ LBP _______________#

	radius = 3
	n_points = 8 * radius

	lbp = local_binary_pattern(image, n_points, radius, method='uniform')
	n_bins = int(lbp.max() + 1)
	hist, _ = np.histogram(lbp, normed = True, bins = n_bins, range =(0, n_bins))
	#Calcule o histograma de um conjunto de dados.
	# O histograma do resultado da LBP é uma boa medida para classificar as texturas.
	carac_img = np.concatenate((carac_img,hist), axis = None)


	
	#_______________ OUTRO _______________#
	#
	#
	#
	#



	escrita = '%s' % (label)
	for i in range(1, len(carac_img)):
		str_aux = ' ' + str(i)+ ":%f" % (carac_img[i])
		escrita = escrita + str_aux
	escrita = escrita + '\n'
	# colocar label+caracteristicas em uma string só

	arq = open('caracteristicas.txt','a')
	arq.write(escrita)
	arq.close(	)