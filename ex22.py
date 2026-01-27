import cv2
import numpy as np

# 1. Load the image in grayscale
path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Define the kernel
    # A 5x5 or 7x7 kernel is often used to bridge larger gaps
    kernel = np.ones((5, 5), np.uint8)

    # 3. Apply the Closing operation
    # This is equivalent to dilate() followed by erode()
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 4. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Closing (Dilation then Erosion)', img_closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()