import cv2
import numpy as np
import time
from rknn.api import RKNN
from utils.rknn_inference import inferenceFunc, CentroidTracker
from collections import deque

IMG_SIZE = (640, 640)

tracker = CentroidTracker()
skip_frames = 1

def main():
    model_path = 'weights/yolo11n.rknn'
    video_path = 'datasets/autos.mp4'
    output_path = 'results/resultado.mp4'

    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(target='rk3588s')

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

    recent_times = deque(maxlen=30) 
    frame_count = 0
    avg_fps_recent = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        if frame_count % skip_frames == 0:
            processed_frame, detections = inferenceFunc(rknn, frame)
            tracked_objects = tracker.update(detections)
        else:
            processed_frame = frame.copy()
            tracked_objects = tracker.objects  # mantener los últimos objetos detectados

        # Dibujar trayectoria
        for object_id, centroid in tracked_objects.items():
            # Línea de trayectoria
            pts = tracker.tracks[object_id]
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(processed_frame, pts[i - 1], pts[i], (0, 255, 255), 2)

            # Círculo e ID
            cv2.circle(processed_frame, centroid, 4, (0, 255, 0), -1)
            cv2.putText(processed_frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        end = time.time()
        elapsed = end - start
        recent_times.append(elapsed)
        frame_count += 1

        if frame_count % 30 == 0:
            avg_fps_recent = 30 / sum(recent_times)
            print("FPS ave: ", avg_fps_recent)

        cv2.putText(processed_frame, f"FPS ave: {avg_fps_recent:.2f}",
                    (processed_frame.shape[1] - cv2.getTextSize(f"FPS ave: {avg_fps_recent:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # out.write(processed_frame)  # Guardar si quieres
        cv2.imshow('RKNN Video Inference', processed_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()
