from abc import ABC, abstractmethod
from biosppy import ecg as bioecg
import numpy as np


class AbstractProcessor(ABC):
    """Abstract wrapper for different processors:
    - Filter - Qrs seeker -
    """
    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def attach(self):
        pass


#################################################### FILTERS ###########################################################
class AbstractFilter(AbstractProcessor):
    def __init__(self, a=None, b=None):
        self.record = None
        self.fs = None
        self.a = a
        self.b = b

    def set_a_b(self, a, b):
        self.a = a
        self.b = b

    def attach(self, record, fs):
        self.fs = fs
        self.record = record

    @abstractmethod
    def process(self):
        """Should return filtered record."""
        pass


class BioSppyFilter(AbstractFilter):
    """Wrapper for Biosspy ecg function, returns filtered ecg record. """
    def process(self):
        out_tuple = bioecg.ecg(np.ravel(self.record), show=False, sampling_rate=self.fs)
        return out_tuple['filtered']


########################################### TEMPLATORS ######################################################
class AbstractTemplator(AbstractProcessor):
    """Abstract class for mean template ECG wave extractors."""
    def __init__(self):
        self.record = None
        self.segments = None
        self.rpeaks = None

    @abstractmethod
    def process(self):
        pass

    def attach(self, record, segments, rpeaks):
        self.record = record
        self.segments = segments
        self.rpeaks = rpeaks


class CorrelationTemplator(AbstractTemplator):
    def process(self, n_segments=8):
        """Extract mean ECG wave of every 'n_segments' waves step by step
        by finding the most correlated 3 btwn them. """
        templates = []
        rpeaks_templates = []
        for segm_idx in range(0, len(self.segments)-n_segments, 2):
            segments_to_check_corr = self.segments[segm_idx:segm_idx+n_segments]
            try:
                correlation = np.corrcoef(segments_to_check_corr)
            except:
                continue
            rows, cols = correlation.shape

            # Since matrix is symmetric, set pair value to 0
            for col in range(cols):
                correlation[col, 0:col+1] = 0

            # Get 1st maximum in matrix
            first_max = np.max(correlation)
            first_max_row, first_max_col = np.where(correlation == first_max)
            correlation[first_max_row, first_max_col] = 0

            # Decide what coef-nt would be next maximum:
            first_potential_maximum = correlation[first_max_row, :].max()
            second_potential_maximum = correlation[:, first_max_col].max()

            # Now we can get it:
            second_maximum = max([first_potential_maximum, second_potential_maximum])
            second_max_row, second_max_col = np.where(correlation == second_maximum)

            # Let's get indexes of most correlated segments within current 5
            indexes_to_built_template = set([first_max_row[0], first_max_col[0],
                                            second_max_row[0], second_max_col[0]])
            chosen_segments = np.array([segments_to_check_corr[indx]
                                        for indx in indexes_to_built_template])
            # Their R peak's coordinates
            chosen_rpeaks = np.array([self.rpeaks[indx]
                                     for indx in indexes_to_built_template])
            # Calculate mean template cycle btwn them
            templates.append(chosen_segments.mean(axis=0))
            rpeaks_templates.append(np.round(chosen_rpeaks.mean()).astype(int))
        return templates, rpeaks_templates
