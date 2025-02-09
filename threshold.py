import cv2
import numpy as np

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')
rows, cols = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
