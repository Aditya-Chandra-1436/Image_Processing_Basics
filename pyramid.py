
import cv2

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')
# Downscale (Gaussian Pyramid)
down = cv2.pyrDown(image)

# Upscale (Laplacian Pyramid)
up = cv2.pyrUp(image)

cv2.imshow('Downscaled Image', down)
cv2.imshow('Upscaled Image', up)
cv2.waitKey(0)
cv2.destroyAllWindows()
