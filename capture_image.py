#!/usr/bin/python
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 as cv
import numpy as np

from cat_detection import DiffDetection


class Detection: 
    def __init__(self, iterations=10):
        self.dd = DiffDetection()
        self.oldImg = np.zeros((1,1),np.uint8)
        self.newImg = np.zeros((1,1),np.uint8)
        self.iterations = iterations

    def run(self):
        #TODO, move to cat_detection and make PiCamera capturing an interfaced camera image getter

        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()

        try:
            print  'Auto calibration.'
            # from https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
            # Set ISO to the desired value
            camera.iso = 400
            # Wait for the automatic gain control to settle
            time.sleep(2)
            # Now fix the values
            camera.shutter_speed = camera.exposure_speed
            camera.exposure_mode = 'off'
            g = camera.awb_gains
            camera.awb_mode = 'off'
            camera.awb_gains = g


            i=0
            
            # allow the camera to warmup
            time.sleep(0.1)
            while i < self.iterations:
	            i = i + 1
	            self.oldImg = self.newImg.copy()
	            # grab an image from the camera
	            non_entrance = False
	            rawCapture = PiRGBArray(camera)
	            camera.capture(rawCapture, format="bgr")
	            self.newImg = rawCapture.array
	            delta = .0
	            time.sleep(2)
	            if i > 1:
	                (diff, delta) = self.dd.differ(self.oldImg, self.newImg,10)
	                EntranceChange = (np.count_nonzero(diff[1,:])+np.count_nonzero(diff[-1,:])) 
	                noEntranceChange = EntranceChange  < 50
	            else:
	                continue
	            timeStr = time.strftime('%y_%m_%d_%H_%M_%S')
	            if delta > 0.03 and delta < 0.18 and noEntranceChange:
	                cv.imwrite('door_' +  timeStr + '_00_before.png', self.oldImg[:,:,2])
	                cv.imwrite('door_' +  timeStr + '_01_after.png', self.newImg[:,:,2])
	                cv.imwrite('door_' +  timeStr + '_02_diff.png', diff)
	            print 'Diff image at time ' + timeStr + ' had ' + str(delta*100.) + '% entrance change: ' + str(EntranceChange)
        finally:
            camera.close()



if __name__ == "__main__":
    while True:
        d = Detection(100)
        d.run()
