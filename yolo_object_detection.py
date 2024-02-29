import cv2
import numpy as np
import json
from yolo_setup import *

def perform_object_detection(image_path, output_path):
    # Load sample image
    image = cv2.imread(image_path)

    # Extract dimensions of the image
    height, width, channels = image.shape

    # Perform blob conversion for the input image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set input to the network
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(output_layers)

    # initialize object counter
    object_count = 0

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # increment object counter
                object_count += 1

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

    # Save annotated image
    cv2.imwrite(output_path, image)

    # generate a json file containing the needed data
    generate_json("data/output_data.json", class_ids, classes)

    # print object count
    print(f"Number of objects detected: {object_count}")


def generate_json(output_path, class_ids, classes):
    # count the number of occurences of each class
    class_count = {}
    for class_id in class_ids:
        class_name =  classes[class_id]
        class_count[class_name] = class_count.get(class_name, 0) + 1

    # convert the class count dictionnary to JSON format
    json_data = json.dumps(class_count, indent=4)

    # write JSON data to file
    with open(output_path, 'w') as json_file:
        json_file.write(json_data)    
# Call object detection function
perform_object_detection("images/highway-traffic.jpg", "images/annotated-image.jpg")
