# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python3 real_time_object_detection.py --prototxt graph.pbtxt --model frozen_inference_graph.pb
# import the necessary packages
# python3 real_time_object_detection.py --prototxt graph.pbtxt --model frozen_inference_graph.pb --num-frames 100 --display 1
#python3 real_time_object_detection.py --num-frames 100 --display 1

# FPS before threading of video: 20.3
# FPS after threading: 19.7
# FPS after threading of webcam: 25 

# FPS after threading of webcam and neural net: 12.5... system monitor shows sporadic bursts..
# have to change the structure, now it is sending and waiting.. need a cascade approach

#Queues are thread-safe and can be shared between threads!! no need to call a function

from __future__ import print_function
from readThread import readFromThread
from deepThread import deepPassThread
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

print("[INFO] starting threaded video stream...")
vs = readFromThread(src=0).start()

print("[INFO] starting up the neural network...")
net = deepPassThread(args["confidence"],args["num_frames"]).start()

fps = FPS().start()

#Feeding waay too fast...
print("[INFO] Running object detection...")
# loop over the frames from the video stream
while fps._numFrames < args["num_frames"]:
	frame = vs.read()
	blob = cv2.dnn.blobFromImage(frame,size=(300, 300), swapRB=True, crop=False)

	net.submitToNet(blob,frame)

	# update the FPS counter
	fps.update()
	

# stop the timer and display FPS information
vs.stop()
print("[DEBUG] reading stopped, waiting for termination of deepthread")
while not net.stopped:
	time.sleep(0.00001)
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

