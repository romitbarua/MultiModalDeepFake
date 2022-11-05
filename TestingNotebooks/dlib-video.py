import cv2
import numpy as np
import matplotlib.pyplot as plt
 

if __name__ == '__main__':
  #video_path = '../data/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00366/00118.mp4'
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(video_path)
  
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  print('Video Properties')
  print('Video FPS: ', cap.get(cv2.CAP_PROP_FPS))

  
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(frame.shape)
    if ret == True:
  
      # Display the resulting frame
      cv2.imshow('Frame',frame)
      #plt.imshow(frame)
  
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  
  # When everything done, release the video capture object
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()