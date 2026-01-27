import cv2
image = cv2.imread(r"D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
