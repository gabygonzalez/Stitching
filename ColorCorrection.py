import cv2
import numpy as np

class color():

    def __init__(self, img):
        self.original = img
        self.correct = self.color(img)

    def color(self, img):
        LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        bgr = cv2.split(HLS)
        b = cv2.equalizeHist(bgr[0])
        g = cv2.equalizeHist(bgr[1])
        r = cv2.equalizeHist(bgr[2])

        #cv2.imshow("lab", cv2.merge((b, g, r)))

        return img
