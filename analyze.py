from abc import abstractmethod
from scipy.signal import welch
from MyModules.ecg_class.process import AbstractProcessor
from biosppy.signals import ecg as bioecg
import numpy as np


########################################### SEEKERS ###############################################################
class AbstractSeeker:
    def __init__(self):
        self.record = None
        self.fs = None
        self.r_coords = []

    @abstractmethod
    def seek(self):
        """Should return samples of R peaks
        and calculated hr."""
        pass

    def attach(self, record, fs):
        self.record = record
        self.fs = fs


class AbstractQrsSeeker(AbstractSeeker):
    @staticmethod
    def calculate_rr(rpeaks):
        """Just calculate RR as a median of
        cumulative diff-ce of RPeaks array."""
        cumdiff = np.diff(rpeaks)
        return int(np.median(cumdiff))

    @abstractmethod
    def seek(self):
        """Should return samples of R peaks
        and calculated hr."""
        pass

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
    def seek(self):
        self.r_coords = bioecg.ecg(self.record, show=False, sampling_rate=self.fs)['rpeaks']
        cycles, rpeaks_relative, median_rr = self.segment_cycles(self.r_coords)
        return self.r_coords.copy(), median_rr, cycles, rpeaks_relative


class EngzeeQrsSeeker(AbstractQrsSeeker):
    """Wrapper for Biosppy Engzee + Lourenco algorithm with threshold 0.48."""
    def seek(self):
        """Locate R peaks and segment ECG cycles."""
        self.r_coords = bioecg.engzee_segmenter(np.ravel(self.record), sampling_rate=self.fs)['rpeaks']
        cycles, rpeaks_relative, median_rr = self.segment_cycles(self.r_coords)
        return self.r_coords.copy(), median_rr, cycles, rpeaks_relative


class PWaveSeeker(AbstractSeeker):
    """Class to detect P-wave and get it's coordinates on a filtered record."""
    def __init__(self):
        super().__init__()
        self.p_to_r = None
        self.q_to_r = None

    def attach(self, record, rpeaks, fs):
        """Attach filtered record and it's rpeaks to the seeker."""
        super().attach(record, fs)
        self.r_coords = rpeaks
        self.p_to_r = int(fs * 0.2)  # 0.2 sec from R - low border of P-w start
        self.q_to_r = 20  # VERY roughly!!!

    def seek(self):
        """Find coordinates of P-peaks on filtered record as coordinates of a local maximum within borders."""
        ppeaks = np.array([])
        for rpeak in self.r_coords:
            lowborder = rpeak - self.p_to_r
            highborder = rpeak - self.q_to_r
            # Bordercases:
            if highborder < 0:
                break
            if lowborder < 0:
                lowborder = 0
            # Get P-wave coord:
            ppeaks = np.append(ppeaks, np.argmax(self.record[lowborder:highborder]) + lowborder)
        return ppeaks


class TWaveSeeker(AbstractSeeker):
    """Class to detect T-wave and get it's coordinates on a filtered record."""
    def __init__(self):
        super().__init__()
        self.r_to_s = None
        self.r_to_t = None

    def attach(self, record, rpeaks, fs):
        super().attach(record, fs)
        self.r_coords = rpeaks
        self.r_to_s = int(fs * 0.11)  # mean duration from R to the end of S-wave
        self.r_to_t = int(fs * 0.28)  # mean duration from R to the end of T-wave

    def seek(self):
        """Find coordinates of T-peaks on filtered record as coordinates of a local maximum within borders."""
        tpeaks = np.array([])
        record_len = len(self.record)
        for rpeak in self.r_coords:
            lowborder = rpeak + self.r_to_s
            highborder = rpeak + self.r_to_t
            # Bordercases:
            if lowborder > record_len:
                break
            if highborder > record_len:
                highborder = record_len
            # Get T-wave coord:
            tpeaks = np.append(tpeaks, np.argmax(self.record[lowborder:highborder]) + lowborder)
        return tpeaks

############################################# ANALYZERS ################################################################
class AbstractWaveAnalyzer(AbstractProcessor):
    """Abstract class for different wave analyzers: f.e. T-wave, QRS-wave, P-wave, ST-segment and so on."""
    def __init__(self):
        self.templates = []  # List of templates
        self.record = []  # Filtered record
        self.rpeak = None  # Coordinate where R-peak located on every template
        self.rpeaks = []  # List of R-peak coordinates on the filtered record
        self.fs = None
        self.p_to_r = None
        self.q_to_r = None

    def attach(self, fs):
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
    """Class to detect P-wave on PQRST templates and extract it's features."""
    def attach(self, templates, rpeak, fs):
        super().attach(fs)
        self.templates = templates
        self.rpeak = rpeak

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
        out_dict['q_coord'] = q_coord
        out_dict['p_coord_rough'] = p_to_q_max_coord
        return out_dict

    def process(self):
        features = []
        for template in self.templates:
            r_to_q = self.cut_off_rq(template, self.rpeak)  # get R to Q segment of the wave
            if r_to_q is None:
                continue
            spectral_features = self.spectral_analyzing(r_to_q)  # get spectral features
            metrical_features = self.metrical_analyzing(template, r_to_q, self.rpeak)  # get metrical features
            features.append((spectral_features, metrical_features))
        return features
