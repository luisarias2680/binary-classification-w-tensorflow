import cv2
import numpy as np



cap = cv2.VideoCapture(r"path")
success,image = cap.read()

fps = cap.get(cv2.CAP_PROP_FPS)
video_length = 62800   
fr = video_length * fps  

n = 15                            
desired_frames = n * np.arange(fr)

for i in desired_frames:
    cap.set(1,i-1)
    success,image = cap.read(1)        
    frameId = cap.get(1)                
    img = image[210:445, 115:350]
    img = cv2.resize(img, (116, 116), interpolation = cv2.INTER_AREA)
    cv2.imwrite(r"...\folder/frame%d.jpg" % frameId, img)

cap.release()
