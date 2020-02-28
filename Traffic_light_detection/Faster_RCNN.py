#importing required libraries
import pandas as pd  
import matplotlib.pyplot as plt 
#matplotlib inline 
from matplotlib import patches
import cv2

#read the csv file using read_csv function of pandas
train = pd.read_csv('../Datasets/BCCD_Dataset-master/test.csv')
train.head()

#reading single image using imread function to matplotlib
image = plt.imread('../Datasets/BCCD_Dataset-master/BCCD/JPEGImages/BloodImage_00000.jpg')
plt.show(image)
cv2.waitKey(0)