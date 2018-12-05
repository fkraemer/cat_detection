import cv2 as cv
import unittest
import numpy as np

class DiffDetection:

    def minTest(self):
        return "la"

    def differ(self, mat1, mat2, threshValue):
        mat1Gray = cv.cvtColor(mat1,cv.COLOR_BGR2GRAY)
        mat2Gray = cv.cvtColor(mat2,cv.COLOR_BGR2GRAY)
        matDiff = cv.absdiff(mat1Gray, mat2Gray)
        (retval, thresh) = cv.threshold(matDiff,threshValue, 255, cv.THRESH_BINARY);

        cnt = np.count_nonzero(thresh)
        total = np.size(thresh,0)*np.size(thresh,1)
        delta = cnt/ float(total)
        print 'Non zero elements are ' + str(delta) + '%'
        return (thresh, delta)



class test(unittest.TestCase):
    # def testImageSetFilling(self):
        # s = 'foo'
        # tmpDir = 'test_tmp'
        # if not os.path.exists(tmpDir):
        #     os.makedirs(tmpDir)
        # fileList = ['img001.png', 'img001_marker.png','img002.png','img003.png', 'img003_marker.png',
        #             'img004_marker.png']
        # for fl in fileList:
        #     filename = '%s/%s' % (tmpDir,fl)
        #     with open(filename, "w") as f:
        #         f.write("FOOBAR")
        # imgBackbone = ImageBackbone(tmpDir,MAX_CLASSES,IMG_SIZE_Y,IMG_SIZE_X,#let the backbone always work up to MAX_CLASSES so that wrong user input can not destroy once set labels
        #                         DEFAULT_BACKGROUND_ALPHA, DEFAULT_MARKER_ALPHA, DEFAULT_WATERSHED_ALPHA,
        #                         MARKER_NEUTRAL_COLOR,MARKER_NEUTRAL_CLASS)
        # #test assertions
        # self.assertEqual(len(imgBackbone.imageSetList),3)
        # imgSetList = imgBackbone.imageSetList
        # self.assertIsNotNone(find_filter(imgSetList,'img001'))
        # self.assertIsNotNone(find_filter(imgSetList,'img003')[0].markerFile)
        # self.assertEqual(len(find_filter(imgSetList,'img004')),0)
        # #clean up
        # for fl in fileList:
        #     os.remove('%s/%s' % (tmpDir,fl))
        # os.removedirs(tmpDir)


    def testMinTest(self):
        s = "la"
        dd = DiffDetection()
        self.assertEqual(s, dd.minTest())