#!/usr/bin/python
# import the necessary packages
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
    rawCapture = PiRGBArray(camera)
    i=0
    # allow the camera to warmup
    time.sleep(0.1)
    while i < self.iterations:
      i = i + 1
      self.oldImg = self.newImg.copy()
      # grab an image from the camera
      print 'Capturing\n'
      camera.capture(rawCapture, format="bgr")
      self.newImage = rawCapture.array
      delta = .0
      ts = time.time()
      if i > 1:
        (diff, delta) = self.dd.differ(self.oldImg, self.newImg)
      if delta > 0.05:
        baseString = 'capture_'+str(ts)
        cv.imwrite(baseString+'_diff.png', diff)
        cv.imwrite(baseString+'_before.png', self.oldImage)
        cv.imwrite(baseString+'_diff.png', self.newImage)
      print 'Diff image at time ' + str(ts) + ' had ' + str(delta) + '%'



if __name__ == "__main__":
    d = Detection()
    d.run()
    
    img1Path = 'image.png'
    img2Path = 'image_before.png'
    mat1 = cv.imread(img1Path, cv.IMREAD_COLOR)
    mat2 = cv.imread(img2Path, cv.IMREAD_COLOR)
    matResult = dd.differ(mat1, mat2, 1)
    cv.imwrite("image_result.png",matResult)
