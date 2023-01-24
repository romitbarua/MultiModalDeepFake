import cv2

def resizeImage(image, height, width, type='reshape', center=None):

    if type=='crop':
        assert not isinstance(center, type(None)), 'If using the crop method, must provide a center tuple'
        pass

    elif type == 'reshape':
        image = cv2.resize(image, (width, height))

def convertColor(image, cv2_color):

    return cv2.cvtColor(image, cv2_color)

    

