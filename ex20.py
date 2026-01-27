import cv2
import numpy as np

# 1. Load the image in grayscale
path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Define the kernel (structuring element)
    # A larger kernel will result in more significant expansion
    kernel = np.ones((5, 5), np.uint8)

    # 3. Apply the Dilation operation
    # iterations=1 defines how many times the operation is repeated
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    # 4. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Dilation Result', img_dilation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()