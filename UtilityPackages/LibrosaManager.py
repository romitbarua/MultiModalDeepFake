import librosa

class LibrosaManager:

    def __init__(self, audio):
        
        self.audio = audio
        
        self.time_series = None
        self.sampling_rate = None 
        
    def _loadLibrosa(self):
        self.time_series, self.sampling_rate = librosa.load(self.audio)


    