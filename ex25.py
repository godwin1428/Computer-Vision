"""
Exercise 25: Recognize Watch from Image using General Object Recognition with OpenCV
Uses pre-trained MobileNet-SSD model for object detection
"""

import cv2
import numpy as np
import urllib.request
import os

# COCO class labels (MobileNet-SSD is trained on COCO dataset)
# Class 85 is 'clock' which includes watches and clocks
CLASSES = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
           "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
           "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
           "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork",
           "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
           "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
           "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]

# Colors for bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def download_model_files():
    """Download pre-trained model files if not present"""
    
    # Model files
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    
    prototxt_path = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    
    if not os.path.exists(prototxt_path):
        print("Downloading prototxt file...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        print("Downloaded prototxt file.")
    
    if not os.path.exists(model_path):
        print("Downloading model file (this may take a while)...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Downloaded model file.")
    
    return prototxt_path, model_path

def recognize_objects(image_path, confidence_threshold=0.5):
    """
    Recognize objects in an image using MobileNet-SSD
    Specifically looks for watch/clock objects
    """
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    (h, w) = image.shape[:2]
    
    # Download model files if needed
    prototxt_path, model_path = download_model_files()
    
    # Load the pre-trained MobileNet-SSD model
    print("Loading MobileNet-SSD model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # MobileNet-SSD classes (VOC dataset - different from COCO)
    VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, 
                                  (300, 300), 127.5)
    
    # Pass the blob through the network
    print("Performing object detection...")
    net.setInput(blob)
    detections = net.forward()
    
    # Store detected objects
    detected_objects = []
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            # Get the class index
            class_idx = int(detections[0, 0, i, 1])
            
            if class_idx < len(VOC_CLASSES):
                class_name = VOC_CLASSES[class_idx]
                
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw bounding box and label
                color = (0, 255, 0) if class_name.lower() in ['clock', 'watch'] else (255, 0, 0)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                
                label = f"{class_name}: {confidence * 100:.2f}%"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (startX, startY, endX, endY)
                })
                
                print(f"Detected: {class_name} with confidence {confidence * 100:.2f}%")
    
    return image, detected_objects

def recognize_watch_cascade(image_path):
    """
    Alternative method using Hough Circle Transform and edge detection
    for watch recognition - optimized for detecting watch faces
    """
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    original = image.copy()
    (h, w) = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply stronger Gaussian blur to reduce noise and mesh pattern interference
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform with stricter parameters
    # Higher param2 = fewer but more confident circles
    # Larger minRadius to avoid small false positives
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,                    # Inverse ratio of resolution
        minDist=100,               # Minimum distance between circle centers
        param1=100,                # Upper threshold for Canny edge detector
        param2=60,                 # Threshold for center detection (higher = stricter)
        minRadius=80,              # Minimum radius - watch face should be large
        maxRadius=250              # Maximum radius
    )
    
    watch_detected = False
    best_circle = None
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Find the most centered/largest circle as the likely watch face
        best_score = -1
        for circle in circles[0, :]:
            center_x, center_y, radius = circle[0], circle[1], circle[2]
            
            # Score based on size and how centered the circle is
            center_dist = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            score = radius - center_dist * 0.5  # Prefer larger, more centered circles
            
            if score > best_score:
                best_score = score
                best_circle = (center_x, center_y, radius)
        
        if best_circle:
            center_x, center_y, radius = best_circle
            
            # Draw the outer circle with thicker line
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Add label with background for better visibility
            label = "Watch Detected"
            label_y = max(center_y - radius - 20, 30)
            cv2.putText(image, label, (center_x - 70, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            watch_detected = True
            print(f"Watch detected at ({center_x}, {center_y}) with radius {radius}")
    
    # Create edge detection output
    edges = cv2.Canny(blurred, 50, 150)
    
    # Only add ellipse detection for significant contours (optional, more selective)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest near-circular contour
    best_ellipse = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Only consider large contours
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                
                # Stricter circularity check
                if minor_axis > 0 and major_axis / minor_axis < 1.3:
                    if area > best_area:
                        best_area = area
                        best_ellipse = ellipse
    
    # Draw only the best ellipse if found and no circle was detected
    if best_ellipse and not watch_detected:
        cv2.ellipse(image, best_ellipse, (255, 0, 255), 3)
        center = best_ellipse[0]
        cv2.putText(image, "Watch Detected", (int(center[0]) - 70, int(center[1]) - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        print(f"Watch detected via ellipse at {center}")
    
    if not watch_detected and not best_ellipse:
        print("No watch detected in the image.")
    
    return image, edges

def main():
    """Main function to demonstrate watch recognition"""
    
    print("=" * 60)
    print("Watch Recognition using OpenCV - General Object Detection")
    print("=" * 60)
    
    # Use the watches_temp.jpg image for watch recognition
    sample_image_path = "sources/watches_temp.jpg"
    
    # Verify the image exists
    if not os.path.exists(sample_image_path):
        print(f"\nError: '{sample_image_path}' not found.")
        print("Please ensure the image file exists in the sources folder.")
        return
    
    print(f"\nUsing image: {sample_image_path}")
    
    # Method 1: Using MobileNet-SSD for general object detection
    print("\n" + "-" * 40)
    print("Method 1: MobileNet-SSD Object Detection")
    print("-" * 40)
    
    try:
        result_image, detected = recognize_objects(sample_image_path)
        if result_image is not None:
            cv2.imshow("MobileNet-SSD Detection", result_image)
            cv2.imwrite("result_mobilenet.jpg", result_image)
            print(f"\nSaved result to 'result_mobilenet.jpg'")
    except Exception as e:
        print(f"MobileNet-SSD detection failed: {e}")
        print("Proceeding with alternative method...")
    
    # Method 2: Using Hough Circles and Edge Detection
    print("\n" + "-" * 40)
    print("Method 2: Circle Detection (Watch Face)")
    print("-" * 40)
    
    result_image2, edges = recognize_watch_cascade(sample_image_path)
    if result_image2 is not None:
        cv2.imshow("Circle Detection - Watch Recognition", result_image2)
        cv2.imshow("Edge Detection", edges)
        cv2.imwrite("result_circle_detection.jpg", result_image2)
        cv2.imwrite("result_edges.jpg", edges)
        print(f"\nSaved results to 'result_circle_detection.jpg' and 'result_edges.jpg'")
    
    print("\n" + "=" * 60)
    print("Press any key to close the windows...")
    print("=" * 60)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nWatch recognition completed!")

if __name__ == "__main__":
    main()
