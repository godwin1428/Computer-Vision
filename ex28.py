import cv2
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained Haar cascade for cars
car_cascade_path = os.path.join(script_dir, "sources", "cars.xml")
car_cascade = cv2.CascadeClassifier(car_cascade_path)

# Check if cascade loaded successfully
if car_cascade.empty():
    print(f"Error: Could not load cascade from {car_cascade_path}")
    exit()

# Load the video file (or use '0' for webcam)
video_path = os.path.join(script_dir, "sources", "traffic.mp4")
video = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frame-by-frame
while True:
    ret, frame = video.read()
    if not ret:
        break  # End of video

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    # Draw rectangles around detected vehicles
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Vehicle Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
