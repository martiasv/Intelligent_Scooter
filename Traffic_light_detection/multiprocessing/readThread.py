from threading import Thread
import cv2
import queue
import imutils
import time

#make daemonic?
#now running from everything, not necessary?

class readFromThread:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        #Initialize the FIFO queue for the frames
        #self.frameQueue = queue.Queue()
        #self.frameQueue.put(self.frame)
    def start(self):
		# start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
		# keep looping infinitely until the thread is stopped
        while True:
			# if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return
			# otherwise, read the next frame from the stream
            (self.grabbed, self.frameGrab) = self.stream.read()
            #self.frameQueue.put(imutils.resize(self.frame, width=300))
            self.frame = imutils.resize(self.frameGrab, width=300)

    def read(self):
		# return the frame most recently read
        return self.frame
        #return self.frameQueue.get()
    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True