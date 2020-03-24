# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python3 real_time_object_detection.py --prototxt graph.pbtxt --model frozen_inference_graph.pb
# import the necessary packages
# threading: python3 real_time_object_detection.py --prototxt graph.pbtxt --model frozen_inference_graph.pb --num-frames 2600 --display 1

# FPS before threading: 20.3
# FPS after threading: 19.7
from __future__ import print_function
from readThread import readFromThread
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
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
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

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow(args["model"], args["prototxt"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
#print("[INFO] starting video stream...")
#vs = cv2.VideoCapture('video.mp4')
#time.sleep(2.0)
#fps = FPS().start()

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter

#Where to look
SRC = 'video.mp4'
threading =1

if threading:
	print("[INFO] sampling THREADED frames from webcam...")
	vs = readFromThread(src=SRC).start()
else:
	# initialize the video stream, allow the cammera sensor to warmup
	print("[INFO] starting video stream...")
	vs = VideoStream(src=SRC).start()
	time.sleep(2.0)

fps = FPS().start()


# loop over the frames from the video stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	start_importim = datetime.datetime.now()
	frame = vs.read()
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

	net.setInput(blob)
	detections = net.forward()
	
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])-1
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()
	

# stop the timer and display FPS information
fps.stop()
vs.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

