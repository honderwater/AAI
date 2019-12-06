# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:23:41 2019

@author: HP
"""

import cv2
import numpy as np
import base_dados_teste_2

#LIGA A CAMARA, QUANDO SE CLICA EM "s" TIRA FOTO
#camera = cv2.VideoCapture(1)
#while True:
#    return_value,image = camera.read()
#    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cv2.imshow('image',gray)
#    if cv2.waitKey(1)& 0xFF == ord('s'):
#        cv2.imwrite('test.jpg',gray)
#        break
#camera.release()
#cv2.destroyAllWindows()
#

# FAZ CROP DA IMAGEM
#image = cv2.imread('test.jpg')
#y=100  # ponto de origem y  
#x=225  # ponto de origem x
#h=250  #altura
#w=190  #largura
#crop = image[y:y+h, x:x+w]
##cv2.imshow('crop', crop)
#cv2.imwrite('crop.jpg',crop)
#cv2.waitKey(0) 
  

#SUPOSTAMENTE FAZ CROP DA IMAGEM AUTOMATICAMENTE
#img = cv2.imread('test.jpg',0)
#ret,thresh = cv2.threshold(img,127,255,0)
#im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
#cnt = contours[0]
#M = cv2.moments(cnt)
#print( M )
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])


#RESIZE DA IMAGEM PARA O TAMANHO DAS SAMPLES DE TREINO
imresize = cv2.imread('crop.jpg')
resize = cv2.resize(imresize, (20,20))
cv2.imwrite('resize.jpg', resize)
cv2.waitKey(0)


#PASSA A IMAGEM 20x20 PARA BINÁRIO (PRETO E BRANCO)
img = cv2.imread('resize.jpg',2)
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)


#PASSA A IMAGEM PARA UMA LINHA E POSTERIORMENTE PARA UM ARRAY PARA SER UM FORMATO QUE O CLASSIFIER ACEITE (NÃO ACEITA MATRIZ)
c = bw_img
resized = c.flatten()
carray = np.array(resized, dtype = np.float32)
#cv2.imshow("cells", array)


#KNN   
knn = cv2.ml.KNearest_create()    
knn.train(base_dados_teste_2.cells, cv2.ml.ROW_SAMPLE, base_dados_teste_2.labels)
result = knn.findNearest(carray, k = 1)
print(result)

#knn.findNearest não está funcionar não sei porquê apesar de estar no formato correto penso eu






















































