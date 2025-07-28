# person_vehicle_detector.py
import torch
import cv2

# Load YOLOv5s model from local repo
model = torch.hub.load('yolov5', 'yolov5s', source='local')

# Define allowed classes
target_labels = ['person', 'car', 'bicycle', 'motorcycle', 'airplane', 'boat']

# Initialize webcam
cap = cv2.VideoCapture(0)

window_name = "People & Vehicle Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    # Check if window is still open
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame by 1.7x
    frame = cv2.resize(frame, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_LINEAR)

    # Object detection
    results = model(frame)

    # Convert results to pandas DataFrame and filter classes
    detections = results.pandas().xyxy[0]
    filtered = detections[detections['name'].isin(target_labels)]

    # Draw filtered results
    for _, row in filtered.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(window_name, frame)

    # Also break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
