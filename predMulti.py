# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt
import glob

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

def predictImage(image_path, detector, class_labels):

    if "*" in image_path:
        images = glob.glob(image_path)
        for i in images:
            predictImage(i, detector, class_labels)
        return

    # 2. Load image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    # 3. Run detection
    boxes, labels, probs = detector.detect(image, 0.5)

    # print(list(zip(labels, probs)))

    if len(labels) == 0:
        print(image_path, "nothing found")

    for (l, p) in zip(labels, probs):
        print(image_path, class_labels[l], p)



    # # 4. draw detected boxes
    # visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
    #
    # # 5. plot
    # plt.imshow(image)
    # plt.show()


if __name__ == '__main__':
    args = argparser.parse_args()

    # 1. create yolo model & load weights
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    labels = config_parser.get_labels()

    for image in args.images:
        predictImage(image, detector, labels)
