import cv2
import numpy as np
import time
from rknn.api import RKNN
from utils.rknn_inference import inferenceFunc
from collections import deque

IMG_SIZE = (640, 640)

def main():
    model_path = 'weights/yolov8n.rknn'
    video_path = 'datasets/autos_short.mp4'
    output_path = 'results/resultado.mp4'

    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(target='rk3588s')

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    #out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

    recent_times = deque(maxlen=30) 
    frame_count = 0
    avg_fps_recent = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        processed_frame = inferenceFunc(rknn,frame)
        end = time.time()

        elapsed = end - start
        recent_times.append(elapsed)
        frame_count += 1
	
        if frame_count % 30 == 0:
            avg_fps_recent = 30 / sum(recent_times)
            print("FPS ave: ",avg_fps_recent)
           
        cv2.putText(processed_frame, f"FPS ave: {avg_fps_recent:.2f}", (processed_frame.shape[1] - cv2.getTextSize(f"FPS ave: {avg_fps_recent:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save video
        #out.write(processed_frame) 

        # Mostrar únicamente (sin guardar)
        cv2.imshow('RKNN Video Inference', processed_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    #out.release() 
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()

