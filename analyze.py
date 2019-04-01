from abc import abstractmethod
from scipy.signal import welch
from MyModules.ecg_class.process import AbstractProcessor
from biosppy.signals import ecg as bioecg
import numpy as np


########################################### QRS SEEKERS ###############################################################
class AbstractQrsSeeker(AbstractProcessor):
    def __init__(self):
        self.record = None
        self.fs = None

    @abstractmethod
    def process(self):
        """Should return samples of R peaks
        and calculated hr."""
        pass

    def attach(self, record, fs):
        self.record = record
        self.fs = fs

    @staticmethod
    def calculate_rr(rpeaks):
        """Just calculate RR as a median of
        cumulative diff-ce of RPeaks array."""
        cumdiff = np.diff(rpeaks)
        return int(np.median(cumdiff))

    def segment_cycles(self, rpeaks):
        """Get median RR interval and extract ECG cycles to an array."""
        cumdiff = np.diff(rpeaks)
        rr = int(np.median(cumdiff))  # Median RR interval
        # Segmenting:
        cycles = []
        lower_limit = 0
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
        return cycles, rpeaks[-1] - lower_limit, rr


class BioSppyQrsSeeker(AbstractQrsSeeker):
    """Wrapper for Biosppy ecg function, returns R peaks coords and HR."""
    def process(self):
        rpeaks = bioecg.ecg(self.record, show=False, sampling_rate=self.fs)['rpeaks']
        cycles, rpeaks_relative, median_rr = self.segment_cycles(rpeaks)
        return rpeaks, median_rr, cycles, rpeaks_relative


class EngzeeQrsSeeker(AbstractQrsSeeker):
    """Wrapper for Biosppy Engzee + Lourenco algorithm with threshold 0.48."""
    def process(self):
        """Locate R peaks and segment ECG cycles."""
        rpeaks = bioecg.engzee_segmenter(self.record, sampling_rate=self.fs)['rpeaks']
        cycles, rpeaks_relative, median_rr = self.segment_cycles(rpeaks)
        return rpeaks, median_rr, cycles, rpeaks_relative


############################################# ANALYZERS ################################################################
class AbstractWaveAnalyzer(AbstractProcessor):
    """Abstract class for different wave analyzers:
    f.e. T-wave, QRS-wave, P-wave, ST-segment and so on.
    """
    def __init__(self):
        self.templates = []
        self.rpeak = None
        self.fs = None
        self.p_to_r = None
        self.q_to_r = None

    def attach(self, templates, rpeak, fs):
        self.templates = templates
        self.rpeak = rpeak
        self.fs = fs
        self.p_to_r = int(fs * 0.2)  # 0.2 sec from R - low border of P-w start
        self.q_to_r = 20  # VERY roughly!!!

    @abstractmethod
    def process(self):
        pass

    def cut_off_rq(self, template, rpeak):
        """Get R-Q segment of the wave."""
        try:
            return template[rpeak - self.p_to_r:rpeak - self.q_to_r]
        except:
            return None


class PWaveAnalyzer(AbstractWaveAnalyzer):
    """Class to detect P-wave and extract some
    features from it.
    """
    def spectral_analyzing(self, r_to_q):
        """Get spectral power density (SPD) and return it's maximum,
        the maximum's coordinate in Hz and area of SPD."""
        f, pxx = welch(r_to_q, fs=self.fs, nperseg=30)
        # Features itself:
        out_dict = dict()
        out_dict['max_pxx'] = np.max(pxx)
        out_dict['max_pxx_coord'] = f[np.argmax(pxx)]  # Get the freq with max Pxx
        out_dict['pxx_area'] = np.trapz(pxx)  # Area under the curve
        # plt.figure()
        # plt.plot(f, Pxx)
        return out_dict

    def metrical_analyzing(self, template, p_to_q, rpeak):
        """Get coordinates of R, Q and return amplitude differences
        btwn R, P, Q, also return sample differences and area under P."""
        p_to_q_max_coord = rpeak - self.p_to_r + np.argmax(p_to_q)  # Rough coordinate of P-wave
        p_range = int(np.round(0.08 * self.fs) / 2)  # Assuming 80ms as P duration, get it's limits
        q_coord = np.argmin(template[p_to_q_max_coord:rpeak]) + p_to_q_max_coord

        # Features itself:
        out_dict = dict()
        out_dict['p_area'] = np.trapz(template[p_to_q_max_coord - p_range:p_to_q_max_coord + p_range])  # And then get an area
        out_dict['p_ampl'] = np.max(p_to_q)  # P amplitude
        out_dict['r_ampl'] = template[rpeak]  # R amplitude
        out_dict['q_ampl'] = template[q_coord]  # Q amplitude

        out_dict['pr_ampl'] = template[rpeak] - template[p_to_q_max_coord]  # Ampl diff-ce btwn R and P
        out_dict['pq_ampl'] = template[p_to_q_max_coord] - template[q_coord]  # Ampl diff-ce btwn P and Q
        out_dict['qr_ampl'] = template[rpeak] - template[q_coord]  # Ampl diff-ce btwn R and Q

        out_dict['pr_samp'] = rpeak - p_to_q_max_coord  # Sample diff-ce btwn R and P
        out_dict['pq_samp'] = q_coord - p_to_q_max_coord  # Sample diff-ce btwn P and Q
        out_dict['qr_samp'] = rpeak - q_coord  # Sample diff-ce btwn R and Q
        return out_dict

    def process(self):
        features = []
        for template in self.templates[:1]:
            r_to_q = self.cut_off_rq(template, self.rpeak)  # get R to Q segment of the wave
            if r_to_q is None:
                continue
            spectral_features = self.spectral_analyzing(r_to_q)  # get spectral features
            metrical_features = self.metrical_analyzing(template, r_to_q, self.rpeak)  # get metrical features
            features.append((spectral_features, metrical_features))
        return np.array(*features)
