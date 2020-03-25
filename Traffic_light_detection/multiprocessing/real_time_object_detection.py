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
#ap.add_argument("-p", "--prototxt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["off","green","yellow","red"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
#print("[INFO] starting video stream...")
#vs = cv2.VideoCapture('video.mp4')
#time.sleep(2.0)
#fps = FPS().start()

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter

#Where to look



print("[INFO] starting threaded video stream...")
vs = readFromThread(src=0).start()

print("[INFO] starting up the neural network...")
net = deepPassThread(args["confidence"],args["num_frames"]).start()

fps = FPS().start()


print("[INFO] Running object detection...")
# loop over the frames from the video stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	start_importim = datetime.datetime.now()
	frame = vs.read()
	#imutils.resize(frame, width=300)
	#if not ret:
	#	print("[INFO] end of video")
	#	break
	# grab the frame dimensions and convert it to a blob
	try:
		(h, w) = frame.shape[:2]
	except:
		break
	blob = cv2.dnn.blobFromImage(frame,size=(300, 300), swapRB=True, crop=False)
	# pass the blob through the network and obtain the detections and
	# predictions

	net.submitToNet(blob,frame)

	# update the FPS counter
	fps.update()
	

# stop the timer and display FPS information
vs.stop()
while not net.stopped:
	time.sleep(0.00001)
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

