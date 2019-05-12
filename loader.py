import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.io.wavfile import read


class AbstractLoader(ABC):
    """Abstract wrapper for loaders from different format files."""
    def __init__(self, init_path, rec_name=None):
        """
        if not isinstance(rec_numb, (int, str, float)) or not str(rec_numb).isdigit():
            raise TypeError('Number of record should be able to be converted to int.')
        """
        if init_path[-2:] != '\\':  # Add special symbols to success loading
            init_path += '\\'

        self.init_path = init_path  # Folder with records
        if rec_name is not None:
            self.rec_name = str(rec_name)  # Concrete record to load
        else:
            self.rec_name = rec_name

    @abstractmethod
    def load_record(self):
        pass

    def set_path(self, path):
        self.init_path = path

    def set_record(self, record):
        self.rec_name = str(record)


class CsvLoader(AbstractLoader):
    """Class to load ECG records from *.csv files."""
    def load_record(self, header='infer', index_col=None):
        """Load record from *.csv to pd.DataFrame."""
        if str(self.rec_name)[-4:] == '.csv':
            whole_path = self.init_path + self.rec_name
            print('Loading...', self.rec_name)
        else:
            whole_path = self.init_path + self.rec_name + '.csv'
            print('Loading...', self.rec_name + '.csv')
        return pd.read_csv(whole_path, header=header, index_col=index_col)


class WavLoader(AbstractLoader):
    """Class to load ECG records from *.wav files."""
    def load_record(self):
        print('Loading...', self.rec_name)
        if str(self.rec_name[-4:]) == '.wav':
            whole_path = self.init_path + self.rec_name
        else:
            whole_path = self.init_path + self.rec_name + '.wav'
        return read(whole_path)
