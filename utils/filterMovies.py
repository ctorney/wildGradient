
import glob
import os
import time
import datetime
import numpy as np
import pandas as pd
import cv2

df = pd.read_csv('../allMovieList.csv')

outname = '../movieList.csv'
outDF = pd.read_csv(outname,index_col=0)

for index,  d in df.iterrows():
    if index<359:
        continue
    movieName = d['filename']

    use = False
    escaped = False
    cap = cv2.VideoCapture(movieName)
    frName = str(index) + ' y use? y or n'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(frName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    stopFrame = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thisFrame = 0
    while True:
        
        if thisFrame>stopFrame-2:
            thisFrame=0
        cap.set(cv2.CAP_PROP_POS_FRAMES,thisFrame)
        _, frame = cap.read()
        thisFrame = thisFrame + 60
                
        cv2.imshow(frName,frame)
        k = cv2.waitKey(10)&0xFF
        
        if k==ord('y'):
            use=True
            break
        if k==ord('n'):
            break
       
        
        if k==27:    # Esc key to stop
            escaped=True
            break 
        
            
    if use:
         outDF.loc[len(outDF)]=[d['filename'],d['mod_date']]
    if escaped:
        break
  
outDF.to_csv(outname)
