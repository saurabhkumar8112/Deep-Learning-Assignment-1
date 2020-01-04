import SimpleITK as sitk 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
array=Image.open("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 1/Test_dataset/normalsJPG/IMG-0001-00001.jpg")
array=np.array(array)
print(array[0])
print(array[0].shape)
#plt.plot(array[0])
plt.imshow(array)
plt.show()

#image_data,image_header=load("C:/Users/Saurabh Kumar/Desktop/ML assignments/ELL888/Assignment 1/BRATS2015_Training/LGG/brats_2013_pat0002_1/VSD.Brain.XX.O.MR_T1.54639/VSD.Brain.XX.O.MR_T1.54639.mha");
#print(image_data.dtype)
