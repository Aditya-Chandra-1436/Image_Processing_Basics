import cv2
import numpy as np
import matplotlib.pyplot as plt

#Step 1: Load the image
image= cv2.imread('full-moon-moon-bright-sky-47367.jpg', cv2.IMREAD_GRAYSCALE)
original = image.copy()

# Step 2: Apply binary Thresholding
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#Step 3: Define KErnel for Morphological operation
kernel = np.ones((5, 5), np.uint8) #5*5 square kernel

# Step4: Perform morphological operations
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

# Step 5: Display Results

plt.figure(figsize=(12, 10))
plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(original, cmap='gray')

plt.subplot(3, 3, 2)
plt.title(f'Binary Image')
plt.imshow(binary, cmap='gray')

plt.subplot(3, 3, 3)
plt.title(f'Erosion')
plt.imshow(erosion, cmap='gray')

plt.subplot(3, 3, 4)
plt.title(f'Dilation')
plt.imshow(dilation, cmap='gray')

plt.subplot(3, 3, 5)
plt.title(f'Opening')
plt.imshow(opening, cmap='gray')

plt.subplot(3, 3, 6)
plt.title(f'Closing')
plt.imshow(closing, cmap='gray')

plt.subplot(3, 3, 7)
plt.title(f'Gradient')
plt.imshow(gradient, cmap='gray')

plt.subplot(3, 3, 8)
plt.title(f'Top Hat')
plt.imshow(tophat, cmap='gray')

plt.subplot(3, 3, 9)
plt.title(f'Black Hat')
plt.imshow(blackhat, cmap='gray')

plt.tight_layout()
plt.show()

#Save Results
cv2.imwrite('erosion.jpg',erosion)
cv2.imwrite('dilation.jpg',dilation)
cv2.imwrite('opening.jpg',opening)
cv2.imwrite('closing.jpg',closing)
cv2.imwrite('gradient.jpg',gradient)
cv2.imwrite('TopHat.jpg',tophat)
cv2.imwrite('BlackHat.jpg',blackhat)

print(f'Morphologicaloperation result saved as separate images')