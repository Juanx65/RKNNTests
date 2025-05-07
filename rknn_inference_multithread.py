import cv2
import numpy as np
from rknnpool import rknnPoolExecutor
from rknn_inference import inferenceFunc

IMG_SIZE = (640, 640)
CONF_THRESH = 0.25
NMS_THRESH = 0.45

TPEs = 3  # Máximo 3 núcleos NPU en RK3588S

def main():
    model_path = 'weights/yolov8n.rknn'
    video_path = 'datasets/autos.mp4'
    output_path = 'results/resultado_video.avi'
    
    pool = rknnPoolExecutor(
        rknnModel=model_path,
        TPEs=TPEs,
        func=inferenceFunc
    )
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter (opcional)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Pre-cargar frames iniciales
    for _ in range(TPEs):
        ret, frame = cap.read()
        if not ret:
            print("No se pudieron leer suficientes frames.")
            cap.release()
            pool.release()
            return
        pool.put(frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Dentro del bucle principal justo antes de pool.put()
            if frame is None or frame.size == 0:
                print("Frame inválido detectado")
                continue
            pool.put(frame)

        processed_frame, flag = pool.get()
        if not flag:
            break
        if processed_frame is None or not isinstance(processed_frame, np.ndarray):
            continue
        #print("Frame recibido:", type(processed_frame), processed_frame.shape)

        # Mostrar resultado
        cv2.imshow('test', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    pool.release()

if __name__ == '__main__':
    main()

