import cv2
from detector import YoloDetector
import time
import os
from deep_sort_realtime.deepsort_tracker import DeepSort


cap =cv2.VideoCapture(0)

forcc = cv2.VideoWriter_fourcc(*'MP4V')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('deepsort.mp4', forcc, 30, (width, height))
detector = YoloDetector()

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

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