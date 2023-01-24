from packages.DlibManager import DlibManager
from packages.LibrosaManager import LibrosaManager
import cv2

class VideoManager:

    def __init__(self, video_path, audio_path=None, phoneme_path=None) -> None:
        
        self.video_path = video_path
        self.video_frames = self.loadVideo()

        self.audio_path = audio_path
        self.phoneme_path = phoneme_path

        self.dlib_manager = DlibManager()
        self.librosa_manager = LibrosaManager(self.audio_path)


    def loadVideo(self):

        video = cv2.VideoCapture(self.video_path)
        frames = []

        while(video.isOpened()):
            ret, frame =  video.read()
            if ret == True:
                frames.append(frame)
            else:
                break
        
        return frames

    def getFacialLandmarks(self, load_object=False):
        return self.dlib_manager._generateLandmarks(self.video_frames, load_object)

    def getLipFrames(self, extension_pixels=20, load_object=False):
        return self.dlib_manager._generateLipFrames(self.video_frames, extension_pixels, load_object)

    def getMFCC(self, n_mfcc=12, window_time=25,  hop_time=10, load_object=False):
        return self.librosa_manager._generateMFCC(n_mfcc = n_mfcc, window_time=window_time, hop_time=hop_time, load_object=load_object)
        


    


    





        
