from json import load
import librosa

class LibrosaManager:

    def __init__(self, audio_path):

        assert not isinstance(audio_path, type(None)), 'Must provide a valid audio path'
        
        self.time_series = None
        self.sampling_rate = None 
        self._loadAudio(audio_path)
        self.duration = librosa.get_duration(self.time_series)

        self.mfcc = None
        
    def _loadAudio(self, audio_path):
        self.time_series, self.sampling_rate = librosa.load(audio_path)

    def _generateMFCC(self, n_mfcc, window_time, hop_time, load_object=False):

        win_length = int(self.sample_rate/1000 * window_time)
        hop_length = int(self.sample_rate/1000 * hop_time)

        mfcc =  librosa.feature.mfcc(self.time_series, n_mfcc=n_mfcc, win_length=win_length, hop_length=hop_length)

        if load_object:
            self.mfcc = mfcc

        return mfcc






    


    