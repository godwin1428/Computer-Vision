import cv2
import numpy as np

# 1. Load the image in grayscale
path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Define the kernel
    # The size of the kernel should be larger than the objects you want to extract
    kernel = np.ones((15, 15), np.uint8)

    # 3. Apply the Top Hat operation
    # Equation: Top Hat = Image - Opening(Image)
    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # 4. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Top Hat Result', img_tophat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()