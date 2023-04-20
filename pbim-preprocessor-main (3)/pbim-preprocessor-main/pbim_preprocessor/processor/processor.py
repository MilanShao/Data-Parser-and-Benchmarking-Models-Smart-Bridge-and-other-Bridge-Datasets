import abc


class Processor(abc.ABC):
    @abc.abstractmethod
    def process(self):
        pass
