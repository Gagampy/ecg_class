from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """Abstract wrapper for different processors:
    - Filter - Qrs seeker -
    """
    @abstractmethod
    def process(self):
        pass
