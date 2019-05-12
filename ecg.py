import numpy as np
from MyModules.ecg_class.loader import CsvLoader, WavLoader


class Ecg:
    """An ecg main class. """
    def __init__(self, loader, rec_filter=None, r_seeker=None, p_seeker=None, t_seeker=None, pw_analyzer=None,
                 visualizator=None, fs=None):
        # Some processor classes
        self.__loader = loader  # takes EcgLoader class obj
        self.__rec_filter = rec_filter  # filter obj
        self.__pw_analyzer = pw_analyzer
        self.__r_seeker = r_seeker  # class to find R coords
        self.__p_seeker = p_seeker  # class to find P coords
        self.__t_seeker = t_seeker  # class to find T coords
        self.__visualizator = visualizator  #
        self.__templator = None

        # Some metainfo
        self.raw_record = None
        self.filtered_record = None
        self.__fs = fs

        # Some ecg's features
        self.r_coords = []
        self.p_coords = []
        self.t_coords = []
        self.__rr = None
        self.templates = []
        self.temp_rpeaks = []
        self.cycles = []
        self.rpeaks_templates_start_coord = None
        self.p_features_spectral = {}
        self.p_features_metric = {}

    def init_record(self, header='infer', index_col=None):
        """Load record via loader object."""
        if type(self.__loader) == CsvLoader:
            self.raw_record = self.__loader.load_record(header=header,
                                                        index_col=index_col)
        if type(self.__loader) == WavLoader:
            self.__fs, self.raw_record = self.__loader.load_record()
        print('Initialization successful...')

    """Some process methods: """
    def filter_record(self, ret=True):
        """Filter the record and return
        filtered signal if ret == True.
        """
        self.filtered_record = self.__rec_filter.process()
        print('Filtered record...')
        if ret:
            return self.filtered_record.copy()

    def seek_r(self, ret=True):
        """Seek R peaks via seeker obj and return their coordinates and estimated HR."""
        out_tuple = self.__r_seeker.seek()  # get array with r coords
        self.r_coords = out_tuple[0]  # R peak coord-tes
        self.__rr = out_tuple[1]  # get RR interval
        self.cycles = out_tuple[2]  # Ecg segments
        self.rpeaks_templates_start_coord = out_tuple[3]  # R coords relative to cycle start
        print('R-peaks found, median rr-interval estimated...')
        if ret:
            return {'rpeaks': out_tuple[0].copy(),
                    'rr': out_tuple[1],
                    'Segments': out_tuple[2].copy(),
                    'r_start': out_tuple[3]}

    def seek_p(self, ret=True):
        """Seek P peaks via seeker obj and return their coordinates."""
        self.p_coords = self.__p_seeker.seek()
        print('P-peaks found...')
        if ret:
            return self.p_coords.copy()

    def seek_t(self, ret=True):
        """Seek T peaks via seeker obj and return their coordinates."""
        self.t_coords = self.__t_seeker.seek()
        print('T-peaks found...')
        if ret:
            return self.t_coords.copy()

    def template(self, ret=True):
        """Create mean templates of ECG
        cycles via templator obj.
        """
        self.templates, self.temp_rpeaks = np.array(self.__templator.process())
        print('Templation is done...')
        if ret:
            return self.templates.copy(), self.temp_rpeaks.copy()

    def pw_analyze(self, ret=True):
        """Extract spectral, metric features characterizing P-wave via P-wave analyzer obj."""
        sm = self.__pw_analyzer.process()
        print('P-wave is analyzed well...')
        if ret:
            return sm

    """Some setters: """
    def set_loader(self, loader=None):
        self.__loader = loader

    def set_r_seeker(self, processor=None):
        if processor is not None:
            self.__r_seeker = processor
        # If seeker wasn't initialized neither here or in __init__:
        if self.__r_seeker is None:
            raise ValueError('R seeker should be initialized by an AbstractFilter inheritor.')
        self.__r_seeker.attach(self.filtered_record, self.__fs)
        print('R seeker is set...')

    def set_p_seeker(self, seeker=None):
        if seeker is not None:
            self.__p_seeker = seeker
        # If seeker wasn't initialized neither here or in __init__:
        if self.__p_seeker is None:
            raise ValueError('P seeker should be initialized by an AbstractFilter inheritor.')
        self.__p_seeker.attach(self.filtered_record, self.r_coords, self.__fs)
        print('P seeker is set...')

    def set_t_seeker(self, seeker=None):
        if seeker is not None:
            self.__t_seeker = seeker
        # If seeker wasn't initialized neither here or in __init__:
        if self.__t_seeker is None:
            raise ValueError('T seeker should be initialized by an AbstractFilter inheritor.')
        self.__t_seeker.attach(self.filtered_record, self.r_coords, self.__fs)
        print('T seeker is set...')

    def set_filter(self, rec_filter=None):
        if rec_filter is not None:
            self.__rec_filter = rec_filter
        # If filter wasn't initialized neither here or in __init__:
        if self.__rec_filter is None:
            raise ValueError('Filter should be initialized by an AbstractFilter inheritor.')
        self.__rec_filter.attach(self.raw_record, self.__fs)
        print('Filter is set...')

    def set_templator(self, templator=None):
        if templator is not None:
            self.__templator = templator
        # If templator wasn't initialized neither here or in __init__:
        if self.__templator is None:
            raise ValueError('Templator should be initialized by an AbstractFilter inheritor.')
        self.__templator.attach(self.filtered_record, self.cycles, self.rpeaks)
        print('Templator is set...')

    def set_pw_analyzer(self, analyzer=None):
        if analyzer is not None:
            self.__pw_analyzer = analyzer
        # If templator wasn't initialized neither here or in __init__:
        if self.__pw_analyzer is None:
            raise ValueError('P-wave analyzer should be initialized by an AbstractFilter inheritor.')
        self.__pw_analyzer.attach(self.templates, self.rpeaks_templates_start_coord, self.__fs)
        print('P-wave analyzer is set...')

    def set_fs(self, fs):
        print('Sample freq-cy is set...')
        self.__fs = fs

    """Some getters: """
    def get_fs(self):
        return self.__fs

    def get_info(self):
        infodict = {'raw': self.raw_record.copy(),
                    'filtered': self.filtered_record.copy(),
                    'fs': self.__fs,
                    'rpeaks': self.rpeaks.copy(),
                    'rr': self.__rr}
        return infodict
