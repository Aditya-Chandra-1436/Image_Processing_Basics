import cv2
import numpy as np
import matplotlib.pyplot as plt

#Step 1: Load the image
image= cv2.imread('red-apple-isolated-clipping-path-19130134.webp')
if image is None:
    raise FileNotFoundError("Image not found ")

#Convert bgr to rgb
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Step 2: Convert to HSV Color Space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Steep 3:
lower_red1 = np.array([0, 100, 100]) #First red tag
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])#Second REd tag
upper_red2 = np.array([180, 255, 255])

#create masks for red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

#Step 4: Apply the mask
segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

#Step 5: Display Original and Segmented Image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Apple')
plt.axis('off')

plt.tight_layout()
plt.show()

#Save Segmented image
cv2.imwrite('segmented_apple.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
print("Segmented image saved as 'segmented_apple.jpg' ")