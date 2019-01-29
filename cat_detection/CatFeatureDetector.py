import cv2 as cv
import unittest
import numpy as np

class CatFeatureDetector:


    def _statisticsFeatures(self, diffBild):
        computeForRegionsN = 3
        image, contours, hierarchy = cv.findContours(diffBild, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        #sort contours to only get the biggest ones
        list.sort(contours, key=len, reverse = True)
        #compute moments for biggest x regions
        featureCnt = 9
        returnArray = np.zeros((computeForRegionsN,featureCnt))
        for i in range(0,min(computeForRegionsN,len(contours))):
            cnt = contours[i]
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #area = cv.contourArea(cnt)
            #perimeter = cv.arcLength(cnt, True)
            huMoments = cv.HuMoments(M)
            returnArray[i,:] = np.concatenate(([cx,cy],huMoments.flatten()))
        #fill, if less were found
        for i in range(min(computeForRegionsN,len(contours)),computeForRegionsN):
            returnArray[i,:] = np.zeros((1,featureCnt))
        return returnArray


    def features(self):
        #aggregate
        aggregatedFeatures = self._statisticsFeatures(self._imgDiff).flatten()

        #return
        return aggregatedFeatures

    def __init__(self, imgBefore, imgAfter, imgDiff):
        self._imgBefore = imgBefore
        self._imgAfter = imgAfter
        self._imgDiff = imgDiff



class test(unittest.TestCase):

    def testFeaturesProduceAnyNonTrivialOutput(self):
        baseImgPath = '/home/flo/data/catpics/positive/capture_1547048777.57_02'
        imgBefore = cv.imread(baseImgPath + '_before.png', cv.IMREAD_GRAYSCALE)
        imgAfter = cv.imread(baseImgPath + '_after.png', cv.IMREAD_GRAYSCALE)
        imgDiff = cv.imread(baseImgPath+'_diff.png',cv.IMREAD_GRAYSCALE)
        dd = CatFeatureDetector(imgBefore, imgAfter, imgDiff)
        returnedFeatures = dd.features()
        self.assertGreater(np.size(returnedFeatures,0),0)
        self.assertFalse(returnedFeatures[0] == 0)
