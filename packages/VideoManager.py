import DlibManager
import LibrosaManager
import cv2

class VideoManager:

    def __init__(self, video_path, audio_path=None, phoneme_path=None) -> None:
        
        self.video_path = video_path
        self.video_frames = self.loadVideo()

        self.audio_path = audio_path
        self.phoneme_path = phoneme_path

        self.dlib_manager = DlibManager(self.video_frames)


    def loadVideo(self):

        video = cv2.VideoCapture(self.video_path)
        frames = []

        while(video.isOpened()):
            ret, frame =  video.read()
            if ret == True:
                frames(frame)
            else:
                break
        
        return frames

    def getFacialLandmarks(self, load_object=False):
        return self.dlib_manager._generateLandmarks(self.video_frames, load_object)

    def getLipFrames(self, extension_pixels=20, load_object=False):
        return self.dlib_manager._generateLipFrames(self.video_frames, extension_pixels, load_object)

    


    





        
