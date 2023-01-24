from lib2to3.pytree import convert
from tkinter import Image
from packages.PcaManager import ImagePcaManager
from packages.Utils.ImageUtils import resizeImage, convertColor
import cv2
import numpy as np

class Featurizer:

    def __init__(self) -> None:
        
        self.X = None
        self.y = None

        self.image_pca_manager = None

    def _setX(self, X):
        self.X = X

    def _setY(self, y):
        self.y = y

    def generatePcaComponents(self, images, fit_pca, n_components, preprocess_data=True, height=None, width=None,
                             resize_type='reshape', center=None):

        if isinstance(self.image_pca_manager, type(None)):
            self.image_pca_manager = ImagePcaManager(n_components)

        if preprocess_data:
            assert not isinstance(height, type(None)), 'If you are preprocessing, you must provide a height'
            assert not isinstance(width, type(None)), 'If you are preprocessing, you must provide a width'

            images = [resizeImage(image, height, width, resize_type, center) for image in images]
            images = [convertColor(image, cv2.COLOR_BGR2GRAY) for image in images]

        images = np.array(images).reshape(len(images), height*width)

        if fit_pca:
            pc = self.image_pca_manager.fit_transform(images)
        else:
            pc = self.image_pca_manager.transform(images)
        
        return pc




        

        

        

    
