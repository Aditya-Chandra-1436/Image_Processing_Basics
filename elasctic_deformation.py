
import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

image = cv2.imread('red-apple-isolated-clipping-path-19130134.webp')

def elastic_transform(image, alpha, sigma):
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

alpha = 34
sigma = 4
elastic_image = elastic_transform(image, alpha, sigma)

cv2.imshow('Elastic Deformation', elastic_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
