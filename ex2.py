import cv2
image = cv2.imread(r"D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg")  # Replace with your image file
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # (15, 15) is the kernel size
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()