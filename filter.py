from abc import abstractmethod
from MyModules.ecg_class.abs_processor import AbstractProcessor
from biosppy import ecg as bioecg
import numpy as np


class AbstractFilter(AbstractProcessor):
    def __init__(self, record, fs, a=None, b=None):
        self.record = record
        self.fs = fs
        self.a = a
        self.b = b

    def set_a_b(self, a, b):
        self.a = a
        self.b = b

    @abstractmethod
    def process(self):
        """Should return filtered record."""
        pass


class BioSppyFilter(AbstractFilter):
    """Wrapper for Biosspy ecg function, returns filtered ecg record. """
    def process(self):
        out_tuple = bioecg.ecg(np.ravel(self.record), show=False, sampling_rate=self.fs)
        return out_tuple['filtered']
