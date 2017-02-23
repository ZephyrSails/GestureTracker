import cv2
import numpy as np


# parameters
cap_region_x_begin = 0.5    # start point/total width
cap_region_y_end = 0.8      # start point/total width
threshold = 60              #  BINARY threshold
blurValue = 41              # GaussianBlur parameter
bgSubThreshold = 16
counter = 0
# variables
isBgCaptured = 1            # bool, whether the background captured


def getMask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur_hsv = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))

    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

    # cv2.imshow('blur', blur_hsv)
    _, thresh = cv2.threshold(blur_hsv, threshold, 255, cv2.THRESH_BINARY)

    densityRange = 10
    pixelDensity = np.zeros_like(thresh)

    edges = cv2.Canny(img, 50, 100)

    edges = 255 - edges

    # mask = cv2.addWeighted(blur_hsv, 0.5, edges, 0.5, 0)
    mask = np.logical_and(blur_hsv, edges) * 255.

    mask = cv2.erode(mask, np.ones((16, 4), np.uint8), iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((40, 20), np.uint8))
    mask = mask.astype(np.uint8)

    # cv2.imshow('edge', edges)
    # cv2.imshow('mask', mask)
    return mask, thresh


def getContours(frame):
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)                      # flip the frame horizontally

    img = frame
    # img = frame[0:int(cap_region_y_end * frame.shape[0]),
    #             int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    mask, thresh = getMask(img)

    """
    Contour threshold?

    Dynamic dilation kernel

    Morphology?

    is montion really useless? - try combine with edge
    """

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, img, thresh


def getRectangle(contours, img, frame):
    res = max(contours, key=cv2.contourArea)
    # res = contours[ci]
    hull = cv2.convexHull(res)
    drawing = np.zeros(img.shape, np.uint8)

    realHandLen = cv2.arcLength(res, True)
    handContour = cv2.approxPolyDP(res, 0.001 * realHandLen, True)
    minX, minY, handWidth, handHeight = cv2.boundingRect(handContour)

    cv2.drawContours(frame, [res], 0, (0, 255, 0), 2)
    cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)

    cv2.rectangle(img, (minX, minY), (minX + handWidth, minY + handHeight), (255, 0, 0))
    return img
