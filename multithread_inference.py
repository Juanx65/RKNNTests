import cv2
import numpy as np
from utils.rknnpool import rknnPoolExecutor
from utils.rknn_inference import inferenceFunc
import time

IMG_SIZE = (640, 640)
TPEs = 3  # Máximo 3 núcleos NPU en RK3588S

def main():
    model_path = 'weights/yolov8n.rknn'
    video_path = 'datasets/autos_short.mp4'
    output_path = 'results/result_multi.mp4'
    
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
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    #out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

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

    frame_count = 0
    total_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame is None or frame.size == 0:
                print("Frame inválido detectado")
                continue
            pool.put(frame)

        start = time.time()
        processed_frame, flag = pool.get()
        end = time.time()

        elapsed = end - start
        total_time += elapsed
        frame_count += 1

        if not flag:
            break
        if processed_frame is None or not isinstance(processed_frame, np.ndarray):
            continue

        avg_fps = frame_count / total_time
        #cv2.putText(processed_frame, f"FPS ave: {avg_fps:.2f}", (processed_frame.shape[1] - cv2.getTextSize(f"FPS ave: {avg_fps:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if frame_count % 30 == 0:
            print("FPS ave: ",avg_fps)
        
        # Save video
        #out.write(processed_frame)
        # Mostrar resultado
        #cv2.imshow('test', processed_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    pool.release()

if __name__ == '__main__':
    main()

