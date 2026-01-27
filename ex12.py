import cv2

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
if img is None:
    print("Error: Image not found")
    exit()

# Step 1: Horizontal flip (simulate y-axis rotation)
flipped = cv2.flip(img, 1)  # 1 → horizontal flip

# Step 2: Rotate 90° clockwise (along z-axis)
rotated_270_y = cv2.rotate(flipped, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('Original', img)
cv2.imshow('270 Degree Y-axis Rotation', rotated_270_y)

cv2.waitKey(0)
cv2.destroyAllWindows()
