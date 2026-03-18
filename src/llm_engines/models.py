from abc import ABC, abstractmethod
from src.config_parser import RunConfig


class EngineProto(ABC):

    def __init__(self, args: RunConfig):
        self.args = args
        
    @abstractmethod
    def load_engine(self):
        pass

    @abstractmethod
    def inference_prompts(self, *args, **kwgs):
        pass

    @abstractmethod
    def shutdown(self):
        pass

