import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolo-coco/yolov3-tiny.weights", "yolo-coco/yolov3-tiny.cfg")
classes = []
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get unconnected output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Initialize colors for drawing bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))
