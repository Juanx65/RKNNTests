import cv2
import numpy as np
import time
from rknn.api import RKNN
from rknn_inference import inferenceFunc

IMG_SIZE = (640, 640)
CONF_THRESH = 0.3
NMS_THRESH = 0.45


def main():
    model_path = 'weights/yolov8m.rknn'
    video_path = 'datasets/autos.mp4'
    output_path = 'datasets/resultado_video.avi'

    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(target='rk3588s')

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = inferenceFunc(rknn,frame)

        # Mostrar únicamente (sin guardar)
        cv2.imshow('RKNN Video Inference', outputs)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    # out.release()  # también eliminar
    cv2.destroyAllWindows()

    rknn.release()

if __name__ == '__main__':
    main()

