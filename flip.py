
import cv2
import numpy as np

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')

# Horizontal Flip
flipped_h = cv2.flip(image, 1)

# Vertical Flip
flipped_v = cv2.flip(image, 0)

# Both Axes Flip
flipped_both = cv2.flip(image, -1)

cv2.imshow('Flipped Horizontally', flipped_h)
cv2.imshow('Flipped Vertically', flipped_v)
cv2.imshow('Flipped Both', flipped_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
