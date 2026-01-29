from io import BytesIO
import logging
from typing import List
from PIL import Image, ImageOps
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results

from entity import BoundingBox

from .port import AbstractObjectDetection


class YoloObjectDetection(AbstractObjectDetection):
    def __init__(self, model_path: str):
        self.device = 'cpu'
        self.model = self._load_model(model_path)
        self.model.to(self.device)

        logging.info(f'Current Device: {self.model.device.__str__()}')

    def _load_model(self, path):
        logging.info('Loading model from pt file ...')

        model = YOLO(path, verbose=False)

        logging.info('Model Loaded ...')

        logging.info('Fusing model ...')

        # model.fuse()

        logging.info('Loading model process complete')

        model.predict('model/Golden_Retriever_Dukedestiny01_drvd.jpg')

        logging.info('Model verify complete')

        return model

    def preprocess(self, input) -> np.ndarray:
        img_io = BytesIO(input)
        img = Image.open(img_io)
        img = ImageOps.exif_transpose(img)
        img_array = np.array(img, dtype=np.float32)
        img.close()

        return img_array

    def inference(self, source) -> Results:
        return self.model.predict(
            source,
            stream=False,
            iou=0.7
        )[0]

    def postprocess(self, input: Results) -> List[dict]:
        results = []

        for box in input.boxes:
            bbox = BoundingBox(
                int(box.cls),
                float(box.conf),
                box.xywhn.tolist().pop()
            ).to_dict()

            results.append(bbox)
            
        return results
