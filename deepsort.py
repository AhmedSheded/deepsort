import cv2
import numpy as np
import sys
import glob
import time
import torch

class YoloDetector():
    def __init__(self, model_name=None):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device: ', self.device)


    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'cusom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1]/downscale_factor)
        height = int(frame.shape[0]/downscale_factor)
        # frame = frame.to(self.device)

        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections=[]

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row=cord[i]
            if row[4]>= confidence:
                x1, y1, x2, y2= int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                if self.class_to_label(labels[i]) == 'person':
                    x_center = x1 + (x2-x1)
                    y_center = y1 + ((y2-y1)/2)

                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = 'person'

                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'person'))

        return frame, detections

cap =cv2.VideoCapture('test3.mp4')

forcc = cv2.VideoWriter_fourcc(*'MP4V')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('deepsort.mp4', forcc, 30, (width, height))
detector = YoloDetector()

#Deep Sort

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from deep_sort_realtime.deepsort_tracker import DeepSort

object_tracker = DeepSort(max_age=5, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)

while cap.isOpened():
    ret, frame = cap.read()

    start = time.perf_counter()
    results = detector.score_frame(frame)
    frame, detections = detector.plot_boxes(results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.4)

    tracks = object_tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb= track.to_ltrb()

        bbox = ltrb
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, 'ID: '+str(track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255))

    end = time.perf_counter()
    totaltime = end-start
    fps = 1/totaltime
    print(fps)
    writer.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1)==27:
        break
cap.release()
writer.release()
cv2.destroyAllWindows()