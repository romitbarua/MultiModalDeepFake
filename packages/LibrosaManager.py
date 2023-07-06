from json import load
import librosa

class LibrosaManager:

    def __init__(self, audio_path):

        assert not isinstance(audio_path, type(None)), 'Must provide a valid audio path' ## SB_Comment - note thi wont gt riggered by an invalid audio path, get sys FileNotFound error '[Errno 2] No such file or directory: 'data'' instead 
        
        self.time_series = None
        self.sampling_rate = None 
        self._loadAudio(audio_path)
        self.duration = librosa.get_duration(y=self.time_series)

        self.mfcc = None
        
    def _loadAudio(self, audio_path):
        self.time_series, self.sampling_rate = librosa.load(audio_path) ## SB_COMMENT: by not specificing sr=None, we are getting their default SR which is 22510, so signal is actually already resampled before we resample it. Suggested improvement - sr=None instead of default arg. This will take the native. 

    def generateMFCC(self, n_mfcc, window_time, hop_time, load_object=False): ## SB_Comment - do we still need this???

        win_length = int(self.sample_rate/1000 * window_time)
        hop_length = int(self.sample_rate/1000 * hop_time)

        mfcc =  librosa.feature.mfcc(self.time_series, n_mfcc=n_mfcc, win_length=win_length, hop_length=hop_length)

        if load_object:
            self.mfcc = mfcc

        return mfcc

    def resample(self, target_sample_rate): ## SB_Comment - noticed some annoyign default behaviours with librosa.load, we need to re-investigate
        return librosa.resample(y=self.time_series.T, orig_sr=self.sampling_rate, target_sr=target_sample_rate)








    


    