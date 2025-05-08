import cv2
import numpy as np
import time
from rknn.api import RKNN
from utils.rknn_inference import inferenceFunc

IMG_SIZE = (640, 640)
CONF_THRESH = 0.3
NMS_THRESH = 0.45

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

    frame_count = 0
    total_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        processed_frame = inferenceFunc(rknn,frame)
        end = time.time()

        elapsed = end - start
        total_time += elapsed
        frame_count += 1

        avg_fps = frame_count / total_time
        #cv2.putText(processed_frame, f"FPS ave: {avg_fps:.2f}", (processed_frame.shape[1] - cv2.getTextSize(f"FPS ave: {avg_fps:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if frame_count % 30 == 0:
            print("FPS ave: ",avg_fps)

        # Save video
        #out.write(processed_frame) 

        # Mostrar únicamente (sin guardar)
        #cv2.imshow('RKNN Video Inference', processed_frame)
        #if cv2.waitKey(1) == 27:
        #    break

    cap.release()
    #out.release() 
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()

