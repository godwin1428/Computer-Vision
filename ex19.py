import cv2
import numpy as np

# 1. Load the image
# Replace 'ex1.jpg' with the full path if the file is not in the script folder
path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded correctly
if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 2. Define the kernel (a 5x5 matrix of ones)
    # Larger kernels or more iterations result in more significant erosion
    kernel = np.ones((5, 5), np.uint8)

    # iterations=1 means the operation is performed once
    img_erosion = cv2.erode(img, kernel, iterations=1)

    # 4. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Erosion Result', img_erosion)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()