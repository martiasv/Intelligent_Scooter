#This is for oing object detection in images. The goal is to later transfer to the real time system.

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from datetime import datetime

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#	"sofa", "train", "tvmonitor"]
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", 
   "street sign","stop sign", "parking meter", "bench", "bird", "cat", "dog", 
   "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe","hat", 
   "backpack", "umbrella", "shoe","eye glasses","handbag", "tie", "suitcase", "frisbee", 
   "skis", "snowboard", "sports ball", "kite", "baseball bat", 
   "baseball glove", "skateboard", "surfboard", "tennis racket", 
   "bottle","plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
   "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
   "hot dog", "pizza", "donut", "cake", "chair", "couch", 
   "potted plant", "bed","mirror", "dining table","window","desk", "toilet", "door","tv", "laptop", 
   "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
   "toaster", "sink", "refrigerator","blender", "book", "clock", "vase", 
   "scissors", "teddy bear", "hair drier", "toothbrush","hair brush"]
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')



# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
#frame = vs.read()
frame = cv2.imread(args["image"])
frame = imutils.resize(frame, width=1600)


# grab the frame dimensions and convert it to a blob
(h, w) = frame.shape[:2]
#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
blob=cv2.dnn.blobFromImage(frame,size=(300, 300), swapRB=True, crop=False)

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
    
    # extract the confidence (i.e., probability) associated with
    # the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > 0.2:
        # extract the index of the class label from the
        # `detections`, then compute the (x, y)-coordinates of
        # the bounding box for the object
        idx = int(detections[0, 0, i, 1])-1
        if labels[idx] == "traffic light":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(labels[idx],confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15 #si el cuadro esta muy arriba pone las
                                            # letras dentro del cuadro, sino fuera 
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            #sobreescribiendo cada classe detectada en un txt
            f = open("class_detected.txt", "w")
            args["image"]
            f.write("{}: {:.2f}%".format(labels[idx],confidence * 100))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            f.write(",Current time: {} and index:{}".format(current_time,idx))
            #if "person"==CLASSES[idx]:
            #	f.write("\nArea del rect: {} - Distancia aprox: {}".format(A,dist))
            f.close()

            #open and read the file after the appending:
            f = open("class_detected.txt", "r")
            print(f.read()) 

# show the output frame
cv2.imshow("Frame", frame)
cv2.waitKey(0)


# do a bit of cleanup
cv2.destroyAllWindows()