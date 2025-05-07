import cv2
import numpy as np
import time
from rknn.api import RKNN
from rknnpool import rknnPoolExecutor
from rknn_inference import inferenceFunc

IMG_SIZE = (640, 640)
CONF_THRESH = 0.3
NMS_THRESH = 0.45

TPEs = 3

def main():
    model_path = 'weights/yolov8n.rknn'
    video_path = 'datasets/autos.mp4'
    output_path = 'results/resultado_video.avi'
    
    pool = rknnPoolExecutor(
	    rknnModel=model_path,
	    TPEs=TPEs,
	    func=inferenceFunc)
	
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

    if (cap.isOpened()):
    	for i in range(TPEs + 1):
    		ret, frame = cap.read()
    		print("Input shape:", frame.shape)  # Debe ser (1, 3, 640, 640)
    		if not ret:
    			cap.release()
    			del pool
    			exit(-1)
    		pool.put(frame)
    
    frames, loopTime, initTime = 0, time.time(), time.time()
    
    while (cap.isOpened()):
    	frames += 1
    	ret, frame = cap.read()
    	if not ret:
    		break
    	pool.put(frame)
    	frame, flag = pool.get()
    	if flag == False:
    		break
    	if frame is None or not isinstance(frame, np.ndarray):
    		continue
    	cv2.imshow('test', frame)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break
    	if frames % 30 == 0:
    		#print("Promedio de FPS en 30 cuadros:\t", 30 / (time.time() - loopTime), "cuadros")
    		loopTime = time.time()
    
    print("Tasa de cuadros promedio total:\t", frames / (time.time() - initTime))

    cap.release()
    # out.release()  # también eliminar
    cv2.destroyAllWindows()
    pool.release()

if __name__ == '__main__':
    main()
