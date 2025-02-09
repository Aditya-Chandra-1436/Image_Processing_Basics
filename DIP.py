import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
import os

# Image PAth
image_path =r'/Users/adityachandra/Desktop/MUSTANG/ ML Training/DIP/55e3d1464356a514f1dc8460962e33791c3ad6e04e507441722973d59745c5_640.jpg'

#1. Check of the image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: Image file not find at the path: {image_path}")

#2. Load the RGB image
rgb_image = cv2.imread(image_path)
if rgb_image is None:
    raise ValueError(f"Error: Unable to load image from path: {image_path}")
rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)

#3. RGB to GRAYSCALE

gray_image= cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray_image.jpg',gray_image)
#cv2.imshow('Grayscale Image', gray_image)
#cv2.waitkey(0)
#cv2.destroyAllWindows() 

#4. Gray to binary
_, binary_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY) # change the threshold value accordingly between 0 to 255
cv2.imwrite('binary_image.jpg', binary_image)

#5 RGB Image Pixel Value
height, width, channels = rgb_image.shape
print(f"Image Dimensions: {width}x{height}, Channels: {channels}")
x, y = 50, 50 # Example pixel coordinates
if x< width and y < height:
    pixel_value = rgb_image[y,x]
    print(f"Pixel Value at({x}, {y}): {pixel_value}")
else:
    print(f"Coordinates ({x}, {y}) are out of image bounds.")

#6. Image HIstogram
plt.figure(figsize=(10,5))
color =('r', 'g', 'b')
for i, col in enumerate(color):
    hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for RGB Image')
plt.xlabel('Pixel Value')
plt.ylabel('frequency')
plt.show()

#7. Pixel Manipuation
for i in range(min(50, height)):
    for j in range(min(50, width)):
        rgb_image[i, j] = [255, 0, 0] #Red color
cv2.imwrite('manipulated_image.jpg',cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

#8. Metadata Extraction
image = Image.open(image_path)
exif_data=image._getexif()
if exif_data is not None:
    print("\n Image Metadata: ")
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        print(f"{tag}: {value}")
else:
    print("/n No metadata found")
print("ALl Tasks completed successfully")