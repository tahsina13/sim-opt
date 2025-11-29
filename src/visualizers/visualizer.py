__all__ = ["Visualizer"]

from abc import ABC, abstractmethod


class Visualizer(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def update(self, k: int):
        pass
