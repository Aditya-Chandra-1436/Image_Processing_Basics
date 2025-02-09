import cv2
import numpy as np

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
