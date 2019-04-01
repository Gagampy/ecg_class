from abc import abstractmethod
import numpy as np
from MyModules.ecg_class.abs_processor import AbstractProcessor


class AbstractTemplator(AbstractProcessor):
    """Abstract class for mean template ECG wave extractors."""
    def __init__(self, record, segments, rpeaks):
        self.record = record
        self.segments = segments
        self.rpeaks = rpeaks

    @abstractmethod
    def process(self):
        pass


class CorrelationTemplator(AbstractTemplator):
    def process(self, n_segments=8):
        """Extract mean ECG wave of every 'n_segments' waves step by step
        by finding the most correlated 3 btwn them. """
        templates = []
        rpeaks_templates = []
        for segm_idx in range(len(self.segments)-n_segments):
            segments_to_check_corr = self.segments[segm_idx:segm_idx+n_segments]
            correlation = np.corrcoef(segments_to_check_corr)
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
            indexes_to_built_template = ([first_max_row[0], first_max_col[0],
                                            second_max_row[0], second_max_col[0]])
            chosen_segments = np.array([segments_to_check_corr[indx]
                                        for indx in indexes_to_built_template])
            # Their R peak's coordinates
            chosen_rpeaks = np.array([self.rpeaks[indx]
                                        for indx in indexes_to_built_template])
            # Calculate mean template cycle btwn them
            templates.append(chosen_segments.mean(axis=0))
            rpeaks_templates.append(chosen_rpeaks.mean(axis=0))
        return templates, rpeaks_templates
