
import dlib
import cv2
import numpy as np


class DlibManager:

    def __init__(self, predictor_path='model/dlib/shape_predictor_68_face_landmarks.dat') -> None:



        self.predictor = dlib.shape_predictor(predictor_path)
        self.detector = dlib.get_frontal_face_detector()

        self.landmarks = None
        self.lip_frames = None

    def _generateLandmarks(self, video_frames, load_object=False):

        video_landmarks = []
        for frame in video_frames:

            #get landmarks
            landmarks = self._generateFrameLandmarks(frame)
            video_landmarks.append(landmarks)

                #overlay landmarks onto image
                #for idx, (x, y) in enumerate(landmarks):
                #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    #cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1,(255,255,255),2,cv2.LINE_AA)
                
                #dlib_frames.append(frame)
        
        if load_object:
            self.landmarks = video_landmarks

        return video_landmarks

    def _generateFrameLandmarks(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.detector(frame_gray)

        landmarks = self.predictor(frame_gray, dets[0])
        landmarks = self._shape_to_np(landmarks)

        return landmarks
        

    def _shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def _generateLipFrames(self, video_frames, extension_pixels, load_object):
        
        lip_landmark_idx = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
        lip_frames = []

        if isinstance(self.landmarks, type(None)):
            landmarks = self._generateLandmarks(video_frames, load_object=False)
        else:
            landmarks = self.landmarks

        for idx, frame in enumerate(video_frames):
            lip_landmarks = landmarks[idx][lip_landmark_idx]

            max_x = np.max(lip_landmarks[:, 0])+extension_pixels
            min_x = np.min(lip_landmarks[:, 0])-extension_pixels
            max_y = np.max(lip_landmarks[:, 1])+extension_pixels
            min_y = np.min(lip_landmarks[:, 1])-extension_pixels

            lip_frame = frame[min_y:max_y, min_x:max_x]
            lip_frames.append(lip_frame)


        if load_object:
            self.lip_frames = lip_frames

        return lip_frames


        









