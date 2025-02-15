import cv2
import numpy as np

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')
rows, cols = image.shape[:2]
# Rotate image by 45 degrees around the center
center = (cols // 2, rows // 2)
M = cv2.getRotationMatrix2D(center, 45, 1)  # Rotate by 45 degrees, scale=1
rotated = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
