import cv2
import numpy as np

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')  # Replace with your image path
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale (optional, dilation works on single-channel)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a kernel (structuring element)
kernel = np.ones((5,5), np.uint8)  # 5x5 kernel of ones

# Apply dilation
dilated = cv2.dilate(gray, kernel, iterations=1)

# Display original and dilated images
cv2.imshow('Original', gray)
cv2.imshow('Dilated', dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()
