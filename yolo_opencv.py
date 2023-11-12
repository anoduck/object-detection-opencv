#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################

import os
import cv2
import random
import string
import numpy as np
from simple_parsing import parse
from dataclasses import dataclass


@dataclass
class HParams:
    idir: str = 'images'
    rdir: str = 'results'
    cnf: str = 'yolov3.cfg'
    wts: str = 'yolov3.weights'
    cls: str = 'yolov3.txt'


Options = parse(HParams)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 10)
    cv2.putText(img, label, (x-10,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)


def get_id(length):
    char = string.ascii_letters + string.digits
    return ''.join(random.choice(char) for _ in range(length))


def main(Options):
    idir = Options.idir
    dir_path = os.path.abspath(idir)
    image_list = os.listdir(dir_path)
    for image_entry in image_list:
        image_file = os.path.join(dir_path, image_entry)
        image = cv2.imread(image_file)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        classes = None
        with open(Options.cls, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        global COLORS
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNet(Options.wts, Options.cnf)
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, classes, class_ids[i],
                            confidences[i],
                            round(x), round(y),
                            round(x+w), round(y+h))
        # cv2.imshow("object detection", image)
        # cv2.waitKey()
        rdir = Options.rdir
        rpath = os.path.abspath(rdir)
        if not os.path.exists(rpath):
            os.makedirs(rpath)
        img_id = get_id(8)
        img_path = os.path.join(rpath, img_id + ".jpg")
        cv2.imwrite(img_path, image)


if __name__ == "__main__":
    main(Options)
