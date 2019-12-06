# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:52:13 2019

@author: HP
"""

import cv2
import numpy as np

digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE) #escala de cinzentos que na realidade é binário...

#digits.png tem 50 linhas e 50 colunas (=2500), cada digito/numero tem 250 amostras para treinar

filas = np.vsplit(digits, 50) #da amostra total digits.png separa em 50 fila, vsplit significa vertical split


cells = [] #necessário separa cada uma das 50 filas em células individuais contendo apenas um número

for fila in filas:
    fila_cells = np.hsplit(fila, 50) #split de cada fila em 50
    for cell in fila_cells:
        cell = cell.flatten()  # necessário para pôr a imagem em uma linha apenas
        cells.append(cell)
        
cells = np.array(cells, dtype = np.float32) #algoritmo apenas funciona com arrays       

# da maneira como foi feito split, cada digito é uma imagem 20x20        
        
k = np.arange(10) # lista de 0 a 9
labels = np.repeat(k, 250)  #dá labels aos números, basicamente dizer que um 0 é um 0
# repete 250 vezes pois na base de dados os digitos estão ordenados e há 250 amostras de cada digito


#KNN
    
knn = cv2.ml.KNearest_create()    
knn.train(cells, cv2.ml.ROW_SAMPLE, labels)
#knn.findNearest(cell_foto, k = 1) # k aqui representa o numero de neighbours

#ret, result, neighbours, dist = knn.findNearest(foto, k = 1)
#print(result)


#cv2.imshow("cells", cells[2499]) #visualizar células individuais
#cv2.imshow("digits", digits) #mostra a base de dados digits

