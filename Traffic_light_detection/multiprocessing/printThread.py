# this thread does all the printing
#from deepThread import deepPassThread
from threading import Thread
import queue
import cv2
import numpy as np


class threadPrinter:
    def __init__(self,globConfidence,nr_frames):
        print("[INFO] initializing printer")
        self.submitQueue = queue.Queue()
        self.frameQueue = queue.Queue()
        self.stopped = False
        self.confidenceThresh = globConfidence
        self.max_nr_frames = nr_frames
        self.frameCtr = 0
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.CLASSES = ["off","green","yellow","red"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def start(self):
        Thread(target=self.livePrinting,args=()).start()
        return self
    def submit(self,detection,frame):
        self.submitQueue.put(detection)
        self.frameQueue.put(frame)

    def livePrinting(self):
        while True:

            if self.stopped or self.frameCtr==self.max_nr_frames:
                self.stopped = True
                cv2.destroyAllWindows()
                return

            if not self.submitQueue.empty():
                detections = self.submitQueue.get()
                frame = self.frameQueue.get()
                (h, w) = frame.shape[:2]
                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > self.confidenceThresh:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])-1
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(self.CLASSES[idx],
                            confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                            self.COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                self.frameCtr = self.frameCtr +1

    def stop(self):
        self.stopped = True
