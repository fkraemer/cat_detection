import unittest
import numpy as np
import cv2 as cv
import math
import os
import random
from CatFeatureDetector import CatFeatureDetector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class CatDetector:
    def __init__(self, trainingSetPositive, trainingSetNegative):
        self._positives = np.array([])
        self._negatives = np.array([])
        self._positiveFileNames = list()
        self._negativeFileNames = list()
        for catPicSet in trainingSetPositive:
            self._negativeFileNames.append(catPicSet._imgBeforePath)
            detector = catPicSet.getFeatureDetector()
            featureArray = detector.features()
            self._positives = featureArray if np.size(self._positives,0) is 0  else np.vstack((self._positives, featureArray ) )
        for catPicSet in trainingSetNegative:
            self._positiveFileNames.append(catPicSet._imgBeforePath)
            detector = catPicSet.getFeatureDetector()
            featureArray = detector.features()
            self._negatives = featureArray if np.size(self._negatives,0) is 0  else np.vstack((self._negatives, featureArray ) )

    def train(self, trainTestSplit):
        positiveSamplesN = np.size(self._positives,0)
        negativeSamplesN = np.size(self._negatives,0)
        labels = np.vstack((np.zeros((negativeSamplesN,1)), np.ones((positiveSamplesN,1)) ) )
        features = np.vstack( (self._negatives, self._positives) )
        self._train_x, self._test_x, self._train_y, self._test_y, self._fileNames_x, self._fileName_y = train_test_split(features, labels, self._negativeFileNames+self._positiveFileNames, train_size=trainTestSplit, random_state=1)    # Train and Test dataset size details
        print "Train_x Shape :: ", self._train_x.shape
        print "Train_y Shape :: ", self._train_y.shape
        print "Test_x Shape :: ", self._test_x.shape
        print "Test_y Shape :: ", self._test_y.shape
        self._clf = RandomForestClassifier(random_state=1, n_estimators=50)
        self._clf.fit(self._train_x, self._train_y.flatten())
        print "Trained model :: ", self._clf

    def test(self):
        self._predictions = self._clf.predict(self._test_x)
        wrongPredictions = np.argwhere(self._test_y.squeeze() != self._predictions)
        if (len(wrongPredictions) > 0 ):
            print "Wrong predictions (%d):\n%s" % (len(wrongPredictions), '\n'.join([self._fileName_y[idx] for idx in wrongPredictions[:,0]]))

        print "Train Accuracy : ", accuracy_score(self._train_y, self._clf.predict(self._train_x))
        print "Test Accuracy  : ", accuracy_score(self._test_y, self._predictions)
        print " Confusion matrix ", confusion_matrix(self._test_y, self._predictions)


    def evalFalsePositives(self): # how often did we predict a cat, when there was none, out of all positive predictions
        cm = confusion_matrix( self._predictions, self._test_y) # y is true label, x is prediction
        return cm[0,1] / float(cm[0,1] + cm[1,1])

    def evalFalseNegatives(self): # how often did we predict no cat, when there was one
        cm = confusion_matrix( self._predictions,self._test_y) # y is true label, x is prediction
        return cm[1,0] / float(cm[0,0] + cm[1,0])



class CatPicSet:

    def __init__(self, imgBeforePath, imgAfterPath, imgDiffPath):
        self._imgBeforePath = imgBeforePath
        self._imgAfterPath = imgAfterPath
        self._imgDiffPath = imgDiffPath

    def getFeatureDetector(self):
        imgBefore = cv.imread(self._imgBeforePath, cv.IMREAD_GRAYSCALE)
        imgAfter = cv.imread(self._imgAfterPath , cv.IMREAD_GRAYSCALE)
        imgDiff = cv.imread(self._imgDiffPath, cv.IMREAD_GRAYSCALE)
        return CatFeatureDetector(imgBefore,
                                  imgAfter,
                                  imgDiff)

class test(unittest.TestCase):

    def setAggregator(self, path):
        picSets = list()
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('02_diff.png'):
                    prefix = filename[:-11]
                    beforeFilename = prefix + '00_before.png'
                    afterFilename = prefix + '01_after.png'
                    if os.path.isfile(path + "/" + beforeFilename) and os.path.isfile(path + "/" + beforeFilename):
                        picSets.append(CatPicSet(path + beforeFilename,
                                                 path + afterFilename,
                                                 path + filename))
        return picSets

    def trainAndClassify(self, randomSeed):
        basePathPositives = '/home/flo/data/catpics/positive/'
        basePathNegatives = '/home/flo/data/catpics/negative/'
        picSetsPositive = self.setAggregator(basePathPositives)
        picSetsNegatives = self.setAggregator(basePathNegatives)
        print 'Sets: %d positive vs. %d negatives' % ( len(picSetsPositive), len(picSetsNegatives) )
        equalSetN = min(len(picSetsPositive), len(picSetsNegatives))
        trainTestSplit = 0.8
        trainingN = int(math.floor(equalSetN * trainTestSplit))
        print 'Balancing. Using %d from each set. %d for training' % ( equalSetN, trainingN )

        #shuffle the list and split into training/test sets
        random.seed(randomSeed)
        random.shuffle(picSetsPositive)
        random.shuffle(picSetsNegatives)

        #train
        cD = CatDetector(picSetsPositive[0:equalSetN], picSetsNegatives[0:equalSetN])
        cD.train(trainTestSplit)
        cD.test()
        falsePositiveScore = cD.evalFalsePositives()
        print 'false positve score is %f' % (falsePositiveScore*100.,)
        falseNegativeScore = cD.evalFalseNegatives()
        print 'false negative score is %f' % (falseNegativeScore*100.,)

        self.assertLess(falsePositiveScore,0.1)
        self.assertLess(falseNegativeScore,0.3)

    def testTrainAndClassifyZero(self):
        self.trainAndClassify(0)
    def testTrainAndClassifyOne(self):
        self.trainAndClassify(1)
    def testTrainAndClassifyTwo(self):
        self.trainAndClassify(2)
    def testTrainAndClassifyThree(self):
        self.trainAndClassify(3)
    def testTrainAndClassifyFour(self):
        self.trainAndClassify(4)