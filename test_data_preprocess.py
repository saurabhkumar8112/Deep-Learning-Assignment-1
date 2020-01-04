import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
test_data=("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 1/Test_dataset")

x_test=[]
y_test=[]

x_test = []
y_test = []

for file in os.listdir(test_data + '/abnormalsJPG'):
    image = cv2.imread(test_data + '/abnormalsJPG/' + file, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    x_test.append(image)
    y_test.append(1)
    
for file in os.listdir(test_data + '/normalsJPG'):
    image = cv2.imread(test_data + '/normalsJPG/' + file, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    x_test.append(image)
    y_test.append(0)
    
x_test = np.asarray(x_test)
print(x_test.shape)
y_test = np.asarray(y_test)
x_train_new=np.load('x_train.npy')
for i in range(len(x_train_new)):
    x_train_new[i]=scale(x_train_new[i])
print(x_train_new.shape)
y_train_new=np.load('y_train.npy')
y_train_new=[0 if y_train_new[i]<=0 else 1 for i in range(len(y_train_new))]
y_train_new=np.array(y_train_new)
y_train_new.reshape((len(y_train_new),1))
x_train_new=x_train_new[0:1000]
y_train_new=y_train_new[0:1000]
print(x_train_new.shape)
x=[]
for i in range(1000):
	x.append(x_train_new[i].reshape((120*120)))
	print(i)
x=np.array(x)
print(x.shape)
'''x=x.reshape((940*x.shape[1]))
for i in range((int)(len(x)/100000)):
	if(x[i]>=1000):
		x[i]=0
	print(i)
x=np.array(x)
x=x[0:1000]
y=y_train_new[0:1000]
print(x.shape)'''

'''pca=TSNE(n_components=2)
projected=pca.fit_transform(x)
print(projected.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color=[ 'r' if(y_train_new[i]==0) else 'b'for i in range(projected.shape[0])]
ax.scatter(projected[:,0],projected[:,1] ,c=color)
plt.show()'''
pca=PCA(2)
projected=pca.fit_transform(x)
print(projected.shape)
color=[ 'r' if(y_train_new[i]==0) else 'b'for i in range(projected.shape[0])]
plt.scatter(projected[:,0],projected[:,1], c=color)
plt.show()





