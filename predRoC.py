# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt
import glob
import json
import sys
import os

from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser

if tf.executing_eagerly():
    print("Executing eargerly")
else:
    print("Executing lazily")

tf.enable_eager_execution()


argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/predict_coco.json",
    help='config file')

argparser.add_argument(
    'images',
    nargs='+',
    help='path to image files')

CAT = []
NOT_CAT = []

def predictImage(image_path, detector, class_labels):

    if "*" in image_path:
        images = glob.glob(image_path)
        for i in images:
            predictImage(i, detector, class_labels)
        return

    global CAT
    global NOT_CAT
    # 2. Load image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    # 3. Run detection
    boxes, labels, probs = detector.detect(image, 0.05)

    # print(list(zip(labels, probs)))

    cat = 0.0

    if len(labels) == 0:
        print(image_path, "nothing found")

    for (l, p) in zip(labels, probs):
        print(image_path, class_labels[l], p)
        if class_labels[l] == "cat":
            cat = max(cat, p)

    is_cat = "not_cat" not in image_path

    if is_cat:
        CAT.append(cat)
    else:
        NOT_CAT.append(cat)

    # # 4. draw detected boxes
    # visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
    #
    # # 5. plot
    # plt.imshow(image)
    # plt.show()

def saveResults():
    global CAT
    global NOT_CAT
    CAT.sort()
    NOT_CAT.sort()
    if len(CAT) == 0:
        print("No cats found")
        return
    if len(NOT_CAT) == 0:
        print("No non-cats found")
        return

    sys.path.append(
        os.path.join(
            os.path.dirname(os.getcwd()),
            "camera"
            )
            )

    import tensorflow1.generate_roc_data
    results = tensorflow1.generate_roc_data.generate_roc_data(CAT, NOT_CAT)
    import json
    open("roc.json","w").write(json.dumps(results))

def main():
    args = argparser.parse_args()

    # 1. create yolo model & load weights
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    labels = config_parser.get_labels()

    for image in args.images:
        predictImage(image, detector, labels)

    saveResults()
    return 0


if __name__ == '__main__':
    sys.exit(main())
