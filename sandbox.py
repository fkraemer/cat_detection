import cv2 as cv
from cat_dection import DiffDetection


if __name__ == "__main__":
    dd = DiffDetection()
    img1Path = 'image.png'
    img2Path = 'image_before.png'
    mat1 = cv.imread(img1Path, cv.IMREAD_COLOR)
    mat2 = cv.imread(img2Path, cv.IMREAD_COLOR)
    threshValue = 7
    (matResult, delta) = dd.differ(mat1, mat2, threshValue)
    cv.imwrite("image_result.png",matResult)
