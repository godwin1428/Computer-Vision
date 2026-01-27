import cv2

img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
if img is None:
    print("Error: Image not found")
    exit()

# Rotate 90 degrees clockwise
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('Original', img)
cv2.imshow('90 Degree Clockwise', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
