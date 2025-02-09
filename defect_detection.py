import cv2
import numpy as np

#step 1: load image
image=cv2.imread('crack.jpeg')
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Step 2: preprocessing
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#step 3: Thresholding
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

#Step 4 : Morphological Operations
kernel = np.ones((5, 5),np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#Step5: Contour Detection
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 100: #flter small noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)

#Step 6: Save the result image
output_path = 'defect_datected_image.jpg'
cv2.imwrite(output_path, image)

#Step 7:Display the image
cv2.imshow('Defect Detection', image)
cv2.waitKey()
cv2.destroyAllWindows()