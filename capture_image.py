#!/usr/bin/python
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 as cv
import numpy as np

from cat_dection import DiffDetection


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
      print 'Capturing\n'
      rawCapture = PiRGBArray(camera)
      camera.capture(rawCapture, format="bgr")
      self.newImg = rawCapture.array
      delta = .0
      time.sleep(2)
      ts = time.time()
      if i > 1:
        if self.oldImg is None:
          print "old img is nore more"
        if self.newImg is None:
          print "new img is nore more"
        (diff, delta) = self.dd.differ(self.oldImg, self.newImg,10)
      else:
        continue
      if delta > 0.05:
        baseString = 'capture_'+str(ts)
        cv.imwrite(baseString+'_00_before.png', self.oldImg[:,:,2])
        cv.imwrite(baseString+'_01_after.png', self.newImg[:,:,2])
        cv.imwrite(baseString+'_02_diff.png', diff)
      print 'Diff image at time ' + str(ts) + ' had ' + str(delta*100.) + '%'



if __name__ == "__main__":
    while True:
        d = Detection(1000)
        d.run()
