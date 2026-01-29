from abc import ABC, abstractmethod
from typing import List

from PIL import Image

import numpy as np


class AbstractObjectDetection(ABC):
    @abstractmethod
    def _load_model(self, path):
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, input) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def inference(self, source) -> any:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, input) -> List[dict]:
        raise NotImplementedError()
        