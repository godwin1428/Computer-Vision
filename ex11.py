import cv2

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
if img is None:
    print("Error: Image not found")
    exit()

# 180-degree rotation along y-axis (horizontal flip)
rotated_y = cv2.flip(img, 1)  # 1 → horizontal flip, -1 → horizontal+vertical

cv2.imshow('Original', img)
cv2.imshow('180 Degree Y-axis Rotation', rotated_y)

cv2.waitKey(0)
cv2.destroyAllWindows()
