import abc
from loguru import logger as logging
import sys

__all__ = [
    'GeneratorRegister',
    'DiscriminatorRegister',
    'LossRegister',
    'OptimizerRegister',
    'SchedulerRegister',
    'PipelineRegister',
    'LayerRegister',
    'MetricRegister',
    'DatasetRegister'
]

class Register(abc.ABC):
    
    def __init__(self, name):
        self.__name__ = name.lower()
        self.__list = {}
    
    def __getitem__(self, name):
        name = name.lower()
        if name not in self.__list:
            logging.error(f"Name {name} not found in {self.__class__.__name__}")
            sys.exit(1) 
        else:
            return self.__list[name]
    
    def __setitem__(self, name, value):
        name = name.lower()
        self.__list[name] = value
    
    def register(self, name):
        def wrapper(cls):
            if name in self.__dict__:
                logging.error(f"Name {name} already registered")
                sys.exit(1)
            self[name] = cls
            return cls
        return wrapper
    
    @property
    def list(self):
        return list(self.__list.keys())
    
class GeneratorRegister(Register):
    def __init__(self):
        super().__init__('generator')

class DiscriminatorRegister(Register):
    def __init__(self):
        super().__init__('discriminator')

class LossRegister(Register):
    def __init__(self):
        super().__init__('loss')

class OptimizerRegister(Register):
    def __init__(self):
        super().__init__('optimizer')

class SchedulerRegister(Register):
    def __init__(self):
        super().__init__('scheduler')

class PipelineRegister(Register):
    def __init__(self):
        super().__init__('pipeline')

class LayerRegister(Register):
    def __init__(self):
        super().__init__('layer')

class MetricRegister(Register):
    def __init__(self):
        super().__init__('metric')

class DatasetRegister(Register):
    def __init__(self):
        super().__init__('dataset')
