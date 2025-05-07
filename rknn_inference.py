import cv2
import numpy as np
import time


IMG_SIZE = (640, 640)
CONF_THRESH = 0.3
NMS_THRESH = 0.45

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def decode_output(original_image, outputs):

    outputs = outputs.transpose(0, 2, 1)  # cambia de (1, 84, 8400) a (1, 8400, 84)
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    h, w = original_image.shape[:2]
    scale_x = w / IMG_SIZE[0]
    scale_y = h / IMG_SIZE[1]

    scale = 1

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= CONF_THRESH:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale_x),
            round(box[1] * scale_y),
             round((box[0] + box[2]) * scale_x),
            round((box[1] + box[3]) * scale_y),
        )
    #cv2.imwrite('resultado.jpg', original_image)
    return original_image, detections

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
   
def inferenceFunc(rknn, frame):
    try:
        assert frame is not None
        assert len(frame.shape) == 3
        # continúa con el preprocesamiento
    except Exception as e:
        print("Error en inferenceFunc:", e)
        return np.zeros((640, 640, 3), dtype=np.uint8)

    start = time.time()
    original_image = frame.copy()
    # Preprocesamiento: redimensionar, convertir a RGB y escalar
    img_resized = cv2.resize(original_image, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, axis=0).astype(np.float32)

    # Inferencia
    outputs = rknn.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        print("❌ Inference failed")
        return frame
        
    end = time.time()
    fps = 1.0 / (end - start)
    cv2.putText(original_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Postprocesamiento con tu función personalizada
    img_out, detections = decode_output(original_image, outputs[0])

    return img_out
