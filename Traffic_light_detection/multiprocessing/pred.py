import numpy as np
import cv2 as cv
import datetime

start_importnet = datetime.datetime.now()
cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
end_importnet = datetime.datetime.now() - start_importnet
print("[TIMESTAMP]: Time elapsed importing model: ")
print( end_importnet.microseconds)

start_importim = datetime.datetime.now()
img = cv.imread('green_close.jpg')
rows = img.shape[0]
cols = img.shape[1]
end_importim = datetime.datetime.now() - start_importim
print("[TIMESTAMP]: Time elapsed importing image: ")
print(end_importim.microseconds)
start_blobify = datetime.datetime.now()
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
end_blobify = datetime.datetime.now() -start_blobify
print("[TIMESTAMP]: Time elapsed blobbifying image: ")
print(end_blobify.microseconds)

start_netpass = datetime.datetime.now()
cvOut = cvNet.forward()
end_netpass = datetime.datetime.now() - start_netpass
print("[TIMESTAMP]: Time elapsed passing image through network: ")
print(end_netpass.microseconds)

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
cv.waitKey()
