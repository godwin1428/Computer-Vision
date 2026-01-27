import cv2
import numpy as np

# 1. Load the image in grayscale
path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Define the kernel
    # Choose a kernel size larger than the dark details you want to extract
    kernel = np.ones((15, 15), np.uint8)

    # 3. Apply the Black Hat operation
    # Equation: Black Hat = Closing(Image) - Image
    img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # 4. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Black Hat Result', img_blackhat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()