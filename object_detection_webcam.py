import numpy as np
import os
import six.moves.urllib as urllib
import urllib.request as allib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import pytesseract
import engineio
import torch
from torch.autograd import Variable as V
import models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import pyttsx3
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

# Initialize text-to-speech engine
engine = pyttsx3.init()

arch = 'resnet18'
model_file = 'whole_%s_places365_python36.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

# Set up Tesseract command
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    tar_file.extractall()
    tar_file.close()
    print('Download complete')
else:
    print('Model already exists')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

url = 'http://10.67.208.240:8080//shot.jpg'

cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        ret = True
        while ret:
            ret, image_np = cap.read()

            if cv2.waitKey(20) & 0xFF == ord('b'):
                cv2.imwrite('opencv' + '.jpg', image_np)

                model_file = 'whole_%s_places365_python36.pth.tar' % arch
                if not os.access(model_file, os.W_OK):
                    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                    os.system('wget ' + weight_url)

                useGPU = 1
                if useGPU == 1:
                    model = torch.load(model_file)
                else:
                    model = torch.load(model_file, map_location=lambda storage, loc: storage)

                model.eval()

                centre_crop = trn.Compose([
                    trn.Resize((256, 256)),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                file_name = 'categories_places365.txt'
                if not os.access(file_name, os.W_OK):
                    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
                    os.system('wget ' + synset_url)
                classes = list()
                with open(file_name) as class_file:
                    for line in class_file:
                        classes.append(line.strip().split(' ')[0][3:])
                classes = tuple(classes)

                img_name = 'opencv.jpg'
                img = Image.open(img_name)
                input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

                logit = model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)

                print('POSSIBLE SCENES ARE: ' + img_name)
                engine.say("Possible Scene may be")
                engine.say(img_name)

                for i in range(0, 5):
                    engine.say(classes[idx[i]])
                    print('{}'.format(classes[idx[i]]))

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            if cv2.waitKey(2) & 0xFF == ord('a'):
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
            else:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
            if cv2.waitKey(2) & 0xFF == ord('r'):
                text = pytesseract.image_to_string(image_np)
                print(text)
                engine.say(text)
                engine.runAndWait()

            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        if apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(image_np, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                print("Warning -Vehicles Approaching")
                                engine.say("Warning -Vehicles Approaching")
                                engine.runAndWait()

                if classes[0][i] == 44:
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        print(apx_distance)
                        engine.say(apx_distance)
                        engine.say("units")
                        engine.say("BOTTLE IS AT A SAFER DISTANCE")

                        if apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(image_np, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                print("Warning -BOTTLE very close to the frame")
                                engine.say("Warning -BOTTLE very close to the frame")
                                engine.runAndWait()

            cv2.imshow('image', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(2) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
