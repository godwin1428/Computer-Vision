import cv2
import numpy as np

# Read the original image
img = cv2.imread(r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\ex1.jpg')
if img is None:
    print("Error: Image not found")
    exit()

# Create a copy to add watermark
watermarked = img.copy()

# Define watermark text
text = "CONFIDENTIAL"
position = (50, 50)  # x, y position
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (0, 0, 255)  # Red color
thickness = 3

# Add semi-transparent watermark
overlay = watermarked.copy()
cv2.putText(overlay, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
alpha = 0.3  # Transparency factor
cv2.addWeighted(overlay, alpha, watermarked, 1 - alpha, 0, watermarked)

# Display results
cv2.imshow('Original', img)
cv2.imshow('Watermarked', watermarked)
cv2.waitKey(0)
cv2.destroyAllWindows()
