import cv2
import numpy as np
from scipy.spatial import distance as dist
import numpy as np

from collections import deque 

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

DARKNET = False
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def decode_output(original_image, outputs):
    outputs = outputs.transpose(0, 2, 1)  # (1, 84, 8400) -> (1, 8400, 84)
    rows = outputs.shape[1]
    boxes, scores, class_ids = [], [], []

    h, w = original_image.shape[:2]
    scale_x = w / IMG_SIZE[0]
    scale_y = h / IMG_SIZE[1]

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= CONF_THRESH:
            box = [
                outputs[0][i][0] - 0.5 * outputs[0][i][2],
                outputs[0][i][1] - 0.5 * outputs[0][i][3],
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, 0.5)
    detections = []

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scaled_box": [
                round(box[0] * scale_x),
                round(box[1] * scale_y),
                round(box[2] * scale_x),
                round(box[3] * scale_y),
            ],
            "scale": 1,
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
    return original_image, detections

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_tracks(frame, tracked_objects, tracker):
    for object_id, centroid in tracked_objects.items():
        color = tracker.colors[object_id]

        # Dibujar trayectoria
        pts = tracker.tracks[object_id]
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(frame, pts[i - 1], pts[i], color, 2)

        # Círculo e ID
        cv2.circle(frame, centroid, 4, color, -1)
        cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class CentroidTracker:
    def __init__(self, max_disappeared=5, max_trace=30):
        self.objects = {}
        self.disappeared = {}
        self.tracks = {}
        self.colors = {}  # ID -> color
        self.next_object_id = 0
        self.max_disappeared = max_disappeared
        self.max_trace = max_trace

    def register(self, centroid, class_id):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = deque(maxlen=self.max_trace)
        self.tracks[self.next_object_id].append(centroid)

        # Usar el color de la clase correspondiente
        self.colors[self.next_object_id] = tuple(map(int, colors[class_id]))

        self.next_object_id += 1



    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.tracks[object_id]
        del self.colors[object_id]


    def update(self, detections):
        input_centroids = []
        input_classes = []
        for det in detections:
            input_centroids.append(self._get_centroid(det["scaled_box"]))
            input_classes.append(det["class_id"])



        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid, class_id in zip(input_centroids, input_classes):
                self.register(centroid, class_id)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.tracks[object_id].append(input_centroids[col])

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_classes[col])


        return self.objects

    def _get_centroid(self, box):
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        return (int(cx), int(cy))

def convert_darknet_to_yolov8_format(output):
    output = np.squeeze(output, axis=0)
    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(-1, 85)
    output = np.expand_dims(output.transpose(1, 0), axis=0)
    return output

def inferenceFunc(rknn, frame):
    original_image = frame.copy()
    img_resized = cv2.resize(original_image, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, axis=0).astype(np.float32)

    outputs = rknn.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        print("❌ Inference failed")
        return frame, []

    if DARKNET:
        ready_output = convert_darknet_to_yolov8_format(outputs[0])
    else:
        ready_output = outputs[0]
    return decode_output(original_image, ready_output)
