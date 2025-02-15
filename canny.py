import cv2
import numpy as np

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)  # Lower and upper threshold

cv2.imshow('Edge Detected Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
