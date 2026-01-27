import cv2
import numpy as np

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
if img is None:
    print("Error: Image not found")
    exit()

rows, cols = img.shape[:2]

# Define 3 points in the original image
pts1 = np.float32([[50,50], [200,50], [50,200]])

# Define corresponding points in the transformed image
pts2 = np.float32([[10,100], [200,50], [100,250]])

# Get the Affine Transformation matrix
M = cv2.getAffineTransform(pts1, pts2)

# Apply the Affine Transformation
dst = cv2.warpAffine(img, M, (cols, rows))

# Display results
cv2.imshow('Original', img)
cv2.imshow('Affine Transformed', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
