from abc import ABC, abstractmethod
from collections.abc import Callable
import threading
from entity import Job, InferenceResponse

class AbstractJobHandler(ABC):
    @abstractmethod
    def register_job_callback(self, callback: Callable[[Job], None]) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def mark_job_as_done(self, results: InferenceResponse) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def set_halt_event(self, event: threading.Event):
        raise NotImplementedError()



        