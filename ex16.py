import cv2
import numpy as np

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel edge detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

# Combine both directions
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

# Display results
cv2.imshow('Original', img)
cv2.imshow('Sobel X', cv2.convertScaleAbs(sobel_x))
cv2.imshow('Sobel Y', cv2.convertScaleAbs(sobel_y))
cv2.imshow('Sobel Combined', sobel_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
