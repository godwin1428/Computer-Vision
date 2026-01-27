import cv2
import numpy as np
image = cv2.imread(r"D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg")  # Replace with your image path
kernel = np.ones((5,5), np.uint8)
eroded_image = cv2.erode(image, kernel, iterations=1)
cv2.imshow("Original Image", image)
cv2.imshow("Eroded Image", eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
