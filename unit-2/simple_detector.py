# simple_object_detector.py
import torch
import cv2

# Load YOLOv5 model from local cloned repo
model = torch.hub.load('yolov5', 'yolov5s', source='local')  # or 'yolov5n' for smaller & faster

# Filter for specific classes
target_labels = ['bottle', 'plate', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple']

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Object Detection")  # Name the window so we can check its property

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Filter results
    filtered = results.pandas().xyxy[0]
    filtered = filtered[filtered['name'].isin(target_labels)]

    # Draw boxes
    for _, row in filtered.iterrows():
        x1, y1, x2, y2, conf, name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Object Detection", frame)

    # Break loop if 'q' is pressed or if window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If window is closed using X button
    if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
