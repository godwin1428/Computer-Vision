import cv2

# Read the image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')  # Replace with your path
if img is None:
    print("Error: Image not found")
    exit()

# Get original dimensions
height, width = img.shape[:2]

# Scale up (enlarge) by 1.5 times
scale_up = cv2.resize(img, (int(width*1.5), int(height*1.5)), interpolation=cv2.INTER_LINEAR)

# Scale down (shrink) by 0.5 times
scale_down = cv2.resize(img, (int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_AREA)

# Display results
cv2.imshow('Original', img)
cv2.imshow('Scaled Up', scale_up)
cv2.imshow('Scaled Down', scale_down)

cv2.waitKey(0)
cv2.destroyAllWindows()
