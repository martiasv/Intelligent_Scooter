from threading import Thread
from printThread import threadPrinter
import cv2
import queue
import time

#start a printing thread


class deepPassThread:
    def __init__(self,confidence,nr_frames):
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
        self.stopped = False
        self.waitQueue = queue.Queue()
        self.doneQueue = queue.Queue()
        self.frameWait = queue.Queue()
        self.frameSub = queue.Queue()
        self.pr = threadPrinter(confidence,nr_frames).start()
    def start(self):
		# start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        #keep looping until thread is stopped
        while True:
            if self.stopped or self.pr.stopped:
                self.stopped = True
                return
            if not self.waitQueue.empty():
            #otherwise, do the things
                self.net.setInput(self.waitQueue.get())
                self.pr.livePrinting(self.net.forward(),self.frameSub.get())
  #def read(self):
        #while self.doneQueue.empty():
        #    time.sleep(0.00001)
    #    return self.doneQueue.get()

    def submitToNet(self,blob,frame):
        self.waitQueue.put(blob)
        self.frameWait.put(frame)


    def stop(self):
        #stop the thread
        self.stopped = True
        self.pr.stop()