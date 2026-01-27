import cv2

# Load the image
image = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')

# Get image dimensions (height, width)
img_h, img_w = image.shape[:2]

# Define ROI (Example: 100x100 area at top-left)
y, x, h, w = 0, 0, 100, 100
roi = image[y:y+h, x:x+w].copy()

# PASTE LOGIC: Let's paste it at the bottom-right instead of a hardcoded 400
# We calculate the start point by subtracting the ROI size from the total image size
dest_y = img_h - h
dest_x = img_w - w

# Destination slicing
image[dest_y:dest_y+h, dest_x:dest_x+w] = roi

cv2.imshow("Corrected Paste", image)
cv2.waitKey(0)
cv2.destroyAllWindows()