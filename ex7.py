import cv2

# Using your specific file path
video_path = r'D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\samplev.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video file. Please check the path.")
    exit()

# Initial delay for Normal Motion (~30 FPS)
delay = 30 

print("--- Video Controls ---")
print("Press 'n' for Normal Motion")
print("Press 's' for Slow Motion")
print("Press 'f' for Fast Motion")
print("Press 'q' to Quit")

while cap.isOpened():
    ret, frame = cap.read()

    # If the video ends, the loop breaks
    if not ret:
        print("End of video reached.")
        break

    # Optional: Resize for better visibility if the video is too large
    # frame = cv2.resize(frame, (800, 450))

    cv2.imshow('Practical: Video Speed Control', frame)

    # Capture keypress
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        delay = 150  # Slow: Increased wait time between frames
        print("Current Mode: Slow Motion")
    elif key == ord('f'):
        delay = 5    # Fast: Decreased wait time between frames
        print("Current Mode: Fast Motion")
    elif key == ord('n'):
        delay = 30   # Normal
        print("Current Mode: Normal Speed")

cap.release()
cv2.destroyAllWindows()