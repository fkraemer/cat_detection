import cv2 as cv
import unittest
import numpy as np

class CatFeatureDetector:


    def statisticsFeatures(self, diffBild):
        return np.asarray([1,2,3])



class test(unittest.TestCase):

    def testFeaturesProduceAnyOutput(self):
        baseImgPath = '/home/flo/data/catpics/positive/capture_1547048777.57_02'
        imgBefore = cv.imread(baseImgPath + '_before.png', cv.IMREAD_GRAYSCALE)
        imgAfter = cv.imread(baseImgPath + '_after.png', cv.IMREAD_GRAYSCALE)
        imgDiff = cv.imread(baseImgPath+'_diff.png',cv.IMREAD_GRAYSCALE)
        dd = CatFeatureDetector()
        returnedFeatures = dd.statisticsFeatures(imgDiff)
        self.assertGreater(np.size(returnedFeatures,0),0)
        self.assertFalse(returnedFeatures[0] == 0)
