#4objectdection.py
#pip install opencv-python numpy
#https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7
#https://github.com/zafarRehan/object_detection_COCO/blob/main/frozen_inference_graph.pb
#install above pb and pbtxt files and extract and give paths below

import numpy as np
import cv2

thres = 0.5
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r"C:\Users\hp\Downloads\production_id_4424421 (2160p).mp4.crdownload")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 850)

classNames = []
with open(r"C:\Users\hp\Downloads\coco.names") as f:
    classNames = f.read().splitlines()

print(classNames)

font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = r"C:\Users\hp\Downloads\frozen_inference_graph.pb"
configPath = r"C:\Users\hp\Downloads\54a8e8b51beb3bd3f770b79e56927bd7-2a20064a9d33b893dd95d2567da126d0ecd03e85 (1)\54a8e8b51beb3bd3f770b79e56927bd7-2a20064a9d33b893dd95d2567da126d0ecd03e85\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

try:
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
except Exception as e:
    print(f"Exception while creating the DetectionModel: {e}")
    import traceback
    traceback.print_exc()
    exit()


net.setInputSize(520, 520)
net.setInputScale(1.0 / 143.5)
net.setInputMean((143.5, 143.5, 143.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            confidence = str(round(confs[i], 2))
            color = Colors[classIds[i] - 1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20), font, 1, color, 2)

    cv2.imshow("output", img)
    cv2.waitKey(1)

