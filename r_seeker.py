from abc import abstractmethod
import numpy as np
from biosppy import ecg as bioecg
from MyModules.ecg_class.abs_processor import AbstractProcessor


class AbstractQrsSeeker(AbstractProcessor):
    def __init__(self, record, fs):
        self.record = record
        self.fs = fs

    @abstractmethod
    def process(self):
        """Should return samples of R peaks
        and calculated hr.
        """
        pass

    def calculate_rr(self, rpeaks):
        """Just calculate RR as a median of
        cumulative diff-ce of RPeaks array"""
        cumdiff = np.diff(rpeaks)
        return int(np.median(cumdiff))

    def segment_cycles(self, rpeaks, rr):
        """Extract ECG cycles to an array."""
        cycles = []
        record_len = len(self.record)
        for r_coord in rpeaks:
            lower_limit = r_coord - int(0.36 * rr)
            upper_limit = r_coord + int(0.64 * rr)
            # Some border conditions
            if lower_limit < 0:
                lower_limit = 0
            if upper_limit > record_len:
                upper_limit = record_len

            cycles.append(self.record[lower_limit:upper_limit])
        return cycles


class BioSppyQrsSeeker(AbstractQrsSeeker):
    """Wrapper for Biosppy ecg function, returns R peaks coords and HR."""
    def process(self):
        rpeaks = bioecg.ecg(np.ravel(self.record), show=False, sampling_rate=self.fs)
        median_rr = self.calculate_rr(rpeaks)
        cycles = self.segment_cycles(rpeaks, median_rr)
        return rpeaks, median_rr, cycles


class EngzeeQrsSeeker(AbstractQrsSeeker):
    """Wrapper for Biosppy Engzee + Lourenco algorithm with threshold 0.48."""
    def process(self):
        """Locate R peaks and segment ECG cycles."""
        rpeaks = bioecg.engzee_segmenter(self.record, sampling_rate=self.fs)['rpeaks']
        median_rr = self.calculate_rr(rpeaks)
        cycles = self.segment_cycles(rpeaks, median_rr)
        return rpeaks, median_rr, cycles
