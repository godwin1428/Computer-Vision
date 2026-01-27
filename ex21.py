import cv2
import numpy as np

# 1. Load the image
path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Define the kernel
    # A 5x5 kernel is standard for general noise removal
    kernel = np.ones((5, 5), np.uint8)

    # 3. Apply the Opening operation
    # This is equivalent to erode() followed by dilate()
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 4. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Opening (Erosion then Dilation)', img_opening)

    cv2.waitKey(0)
    cv2.destroyAllWindows()