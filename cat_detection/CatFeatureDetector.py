import cv2 as cv
import unittest
import numpy as np

class CatFeatureDetector:


    def _statisticsFeatures(self, diffImg):
        computeForRegionsN = 3
        image, contours, hierarchy = cv.findContours(diffImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
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

    def _boundaryChangedFeatures(self, diffImg):
        boundary = .2
        height = np.size(diffImg,0)
        width = np.size(diffImg,1)
        heightBoundaryPx = int(boundary * height)
        heightBoundaryRegionCount = heightBoundaryPx * width
        widthBoundaryPx = int(boundary * width)
        widthBoundaryRegionCount = widthBoundaryPx * height

        upperRegionChanged = np.count_nonzero(diffImg[0:heightBoundaryPx,:]) / float(heightBoundaryRegionCount)
        lowerRegionChanged = np.count_nonzero(diffImg[-heightBoundaryPx:height,:]) / float(heightBoundaryRegionCount)
        leftRegionChanged = np.count_nonzero(diffImg[:,0:widthBoundaryPx]) / float(widthBoundaryRegionCount)
        rightRegionChanged = np.count_nonzero(diffImg[:,-widthBoundaryPx:width]) / float(widthBoundaryRegionCount)
        centralRegionChanged = np.count_nonzero(diffImg[heightBoundaryPx:-heightBoundaryPx,widthBoundaryPx:-widthBoundaryPx]) / float( (height-2*heightBoundaryPx)*(width-2*widthBoundaryPx) )

        return np.asarray([upperRegionChanged, lowerRegionChanged, leftRegionChanged, rightRegionChanged, centralRegionChanged])

    def _colorFeaturesChangedRegions(self, diffImg, beforeImg, afterImg):
        changedIdx = np.nonzero(diffImg)
        colorDiffImg = (np.copy(beforeImg)/2).astype(np.int8)
        colorDiffImg = colorDiffImg - afterImg/2
        changedPx = colorDiffImg[changedIdx]
        hist, binEdges = np.histogram(changedPx,bins=range(-125,150,25))
        hist = hist.astype(np.float) / (len(changedPx)+0.0001)
        return np.asarray(hist)


    def features(self):
        #aggregate
        aggregatedFeatures = np.asarray([])
        #aggregatedFeatures = np.hstack((aggregatedFeatures, self._statisticsFeatures(self._imgDiff).flatten()) )
        aggregatedFeatures = np.hstack((aggregatedFeatures, self._boundaryChangedFeatures(self._imgDiff)))
        aggregatedFeatures = np.hstack((aggregatedFeatures, self._colorFeaturesChangedRegions(self._imgBefore, self._imgAfter, self._imgDiff)))

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
