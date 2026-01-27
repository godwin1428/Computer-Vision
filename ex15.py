import cv2
import numpy as np

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
cv2.imshow('Original', img)
if img is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Result is dilated for marking the corners
dst = cv2.dilate(dst, None)

# Threshold for detecting strong corners
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

# Display the image with corners
cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
