from abc import ABC

class AbstractMQConnector(ABC):
    def get_client(self):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()

    def _connect(self):
        raise NotImplementedError()
