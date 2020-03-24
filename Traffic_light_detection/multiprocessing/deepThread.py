from threading import Thread
import cv2
import queue
import time

class deepPassThread:
    def __init__(self):
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
        self.stopped = False
        self.waitQueue = queue.Queue()
        self.doneQueue = queue.Queue()
    
    def start(self):
		# start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        #keep looping until thread is stopped
        while True:
            if self.stopped:
                return
            if not self.waitQueue.empty():
            #otherwise, do the things
                self.net.setInput(self.waitQueue.get())
                self.doneQueue.put(self.net.forward())
    def read(self):
        #while self.doneQueue.empty():
        #    time.sleep(0.00001)
        return self.doneQueue.get()

    def submitToNet(self,blob):
        self.waitQueue.put(blob)

    def stop(self):
        #stop the thread
        self.stopped = True