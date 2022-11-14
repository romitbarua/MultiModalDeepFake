
import dlib
import cv2
import numpy as np


class DlibManager:

    def __init__(self, predictor, detector, video = None, video_path=None) -> None:

        assert (video is not None and video_path is None) or (video is None and video_path is not None), 'Please provide either an image path or image'

        if video_path is not None:
            self.video_path = video_path
            self.video = self.loadVideoPath()
        else:
            self.video = video

        self.predictor = predictor
        self.detector = detector

        self.full_frames = self._openFrames()
        self.landmarks = self._generateLandmarks()
        self.lip_frames = self._generateLipFrames()

    def _generateLandmarks(self):

        video_landmarks = []
        for frame in self.full_frames:

            #get landmarks
            landmarks = self._generateFrameLandmarks(frame)
            video_landmarks.append(landmarks)

                #overlay landmarks onto image
                #for idx, (x, y) in enumerate(landmarks):
                #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    #cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1,(255,255,255),2,cv2.LINE_AA)
                
                #dlib_frames.append(frame)
        return video_landmarks

    def _generateFrameLandmarks(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.detector(frame_gray)

        landmarks = self.predictor(frame_gray, dets[0])
        landmarks = self._shape_to_np(landmarks)

        return landmarks
        

    def loadVideoPath(self):
        pass

    def _shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def _openFrames(self):
        
        frames = []
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if ret == True:
                frames.append(frame)
            else:
                break

        return frames

    def _generateLipFrames(self, extension_pixels=20):
        
        lip_landmark_idx = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
        lip_frames = []

        for idx, frame in enumerate(self.full_frames):
            lip_landmarks = self.landmarks[idx][lip_landmark_idx]

            max_x = np.max(lip_landmarks[:, 0])+extension_pixels
            min_x = np.min(lip_landmarks[:, 0])-extension_pixels
            max_y = np.max(lip_landmarks[:, 1])+extension_pixels
            min_y = np.min(lip_landmarks[:, 1])-extension_pixels

            lip_frame = frame[min_y:max_y, min_x:max_x]
            lip_frames.append(lip_frame)

        return lip_frames


        









