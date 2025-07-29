import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Load labels for COCO classes
LABELS = {
    1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane', 6: 'Bus',
    7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic Light', 11: 'Fire Hydrant',
    13: 'Stop Sign', 14: 'Parking Meter', 15: 'Bench', 16: 'Bird', 17: 'Cat',
    18: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cow', 22: 'Elephant', 23: 'Bear',
    24: 'Zebra', 25: 'Giraffe', 27: 'Backpack', 28: 'Umbrella', 31: 'Handbag',
    32: 'Tie', 33: 'Suitcase', 34: 'Frisbee', 35: 'Skis', 36: 'Snowboard',
    37: 'Sports Ball', 38: 'Kite', 39: 'Baseball Bat', 40: 'Baseball Glove',
    41: 'Skateboard', 42: 'Surfboard', 43: 'Tennis Racket', 44: 'Bottle',
    46: 'Wine Glass', 47: 'Cup', 48: 'Fork', 49: 'Knife', 50: 'Spoon',
    51: 'Bowl', 52: 'Banana', 53: 'Apple', 54: 'Sandwich', 55: 'Orange',
    56: 'Broccoli', 57: 'Carrot', 58: 'Hot Dog', 59: 'Pizza', 60: 'Donut',
    61: 'Cake', 62: 'Chair', 63: 'Couch', 64: 'Potted Plant', 65: 'Bed',
    67: 'Dining Table', 70: 'Toilet', 72: 'TV', 73: 'Laptop', 74: 'Mouse',
    75: 'Remote', 76: 'Keyboard', 77: 'Cell Phone', 78: 'Microwave',
    79: 'Oven', 80: 'Toaster', 81: 'Sink', 82: 'Refrigerator', 84: 'Book',
    85: 'Clock', 86: 'Vase', 87: 'Scissors', 88: 'Teddy Bear', 89: 'Hair Drier',
    90: 'Toothbrush'
}

# Load the pre-trained model
model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
detector = hub.load(model_url)

# Load image
image_path = "resources/salad.jpg"  # Change this if needed
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
input_tensor = tf.image.convert_image_dtype(input_tensor, tf.float32)
input_tensor = tf.image.resize(input_tensor, (512, 512))[tf.newaxis, :]

# Inference
result = detector(input_tensor)
result = {k: v.numpy() for k, v in result.items()}

# Threshold
threshold = 0.4
count_map = {}

# Draw boxes and count items
for i in range(len(result["detection_scores"])):
    score = result["detection_scores"][i]
    if score < threshold:
        continue

    box = result["detection_boxes"][i]
    class_id = int(result["detection_classes"][i])
    label = LABELS.get(class_id, "Unknown")

    count_map[label] = count_map.get(label, 0) + 1

    ymin, xmin, ymax, xmax = box
    (left, top) = (int(xmin * image.shape[1]), int(ymin * image.shape[0]))
    (right, bottom) = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))

    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(image, f"{label} ({score:.2f})", (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show image
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print count summary
print("\nSummary of Detected Items:")
for item, count in count_map.items():
    print(f"{item}: {count}")
