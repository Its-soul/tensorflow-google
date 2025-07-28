import torch
import cv2

# Load YOLOv5 model from local cloned repo
model = torch.hub.load('yolov5', 'yolov5s', source='local')  # or 'yolov5n' for smaller & faster

# Filter for specific classes
target_labels = ['bottle', 'plate', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple']

# Define window name constant
WINDOW_NAME = "Object Detection"

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Filter results
    filtered = results.pandas().xyxy[0]
    filtered = filtered[filtered['name'].isin(target_labels)]

    # Draw boxes and labels
    for _, row in filtered.iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        x2, y2 = int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        name = row['name']

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence
        label = f'{name} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow(WINDOW_NAME, frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Break if window is closed using the X button
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
