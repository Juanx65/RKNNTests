from queue import Queue
from threading import Thread
from rknnlite.api import RKNNLite


def initRKNN(rknnModel="./rknnModel/yolov5s.rknn", core_id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)

    core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    core_mask = core_masks[core_id % 3]

    ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)

    print(f"{rknnModel} initialized on core {core_id}")
    return rknn_lite


class RKNNWorker(Thread):
    def __init__(self, rknn_index, func, input_queue, output_queue, rknnPool):
        super().__init__()
        self.rknn_index = rknn_index
        self.func = func
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.rknnPool = rknnPool
        self._stop_flag = False
        self.start()

    def run(self):
        while True:
            frame = self.input_queue.get()
            if frame is None:
                break
            result = self.func(self.rknnPool[self.rknn_index], frame)
            self.output_queue.put(result)
            self.input_queue.task_done()


class rknnPoolExecutor:
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = min(TPEs, 3)
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.rknnPool = [initRKNN(rknnModel, i) for i in range(self.TPEs)]
        self.workers = [
            RKNNWorker(i, func, self.input_queue, self.output_queue, self.rknnPool)
            for i in range(self.TPEs)
        ]

    def put(self, frame):
        self.input_queue.put(frame)

    def get(self):
        try:
            result = self.output_queue.get(timeout=1)  # espera hasta 1 segundo
            return result, True
        except:
            return None, False

    def release(self):
        for _ in self.workers:
            self.input_queue.put(None)
        for w in self.workers:
            w.join()
        # ❗️Importante: liberar los RKNN solo después de que todos los hilos terminaron
        for rknn in self.rknnPool:
            rknn.release()

