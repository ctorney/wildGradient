
import glob
import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import cv2
from utils import panZoom
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
DATAHOME = '/home/ctorney/data/tz-2017/'
CODEHOME = '/home/ctorney/workspace/wildGradient/'
inputname = CODEHOME + '/irMovieList.csv'
dfMovies = pd.read_csv(inputname,index_col=0)





def allocate(y,x):#event,x,y,flags,param):
    global counter
 #   if event == cv2.EVENT_LBUTTONDBLCLK:
    
 #   outputArray[counter,0:3] =  cleanFrame[int(y),int(x),:]
 #   outputArray[counter,3]=allocation
    if allocation==WILDEBEEST:
        save_path = "training/yes/img_" + str(counter) + ".png"
    else:
        save_path = "training/no/img_" + str(counter) + ".png"
    grabSize = 16#math.ceil((100.0/alt)*16.0)
    tmpImg =  cleanFrame[max(0,y-grabSize):min(ny,y+grabSize), max(0,x-grabSize):min(nx,x+grabSize)].copy()

            
    if tmpImg.size == 4*grabSize*grabSize*3:# and tmpImg[tmpImg==0].size<10 :
        cv2.imwrite(save_path,tmpImg)
            
            
    counter += 1

def blockUp(y,x):
    global counter
    startX = int(min(x,blockSelectX))
    stopX = int(max(x,blockSelectX))

    startY = int(min(y,blockSelectY))
    stopY = int(max(y,blockSelectY))
    step = 32
    grabSize = 16#math.ceil((100.0/alt)*16.0)
    for i in range(startX,stopX,step):
        for j in range(startY,stopY,step):
            tmpImg =  cleanFrame[max(0,j-grabSize):min(ny,j+grabSize), max(0,i-grabSize):min(nx,i+grabSize)].copy()
            if allocation==WILDEBEEST:
                save_path = "training/yes/img_" + str(counter) + ".png"
            else:
                save_path = "training/no/img_" + str(counter) + ".png"

            
            print(tmpImg.size,'===')
            if tmpImg.size == 4*grabSize*grabSize*3:# and tmpImg[tmpImg==0].size<10 :
                cv2.imwrite(save_path,tmpImg)

            print(counter)
            counter += 1



def blockDown(y,x):
    global blockSelectX
    global blockSelectY
    blockSelectX=x
    blockSelectY=y

blockSelectX=0
blockSelectY=0


    
counter = random.randint(0,10000)
NOTHING=0
WILDEBEEST=1
ZEBRA=2


for index,  d in dfMovies.iterrows():

    filename = d['filename']
    filename = '/media/ctorney/gamma/day-7/DJI_0064.MP4'
    

    cap = cv2.VideoCapture(filename)
    frCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    endClass = False
    for tt in range(frCount):
    
        _, frame = cap.read()
        ny,nx,_ = frame.shape
        if frame is None:
            break
        
        if (tt%60)!=0:
            continue
        
        cleanFrame = np.copy(frame)
        
        #toggle[:,:,:]=0
    
        cv2.destroyAllWindows()
        allocation=WILDEBEEST
        frName = 'click wildebeest (ESC to quit, c to continue)'
  #      cv2.namedWindow(frName, flags =  cv2.WINDOW_GUI_EXPANDED )
  #      cv2.setMouseCallback(frName,is_wildebeest)
  #      cv2.imshow(frName,frame)
        window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=allocate)
        while(1):
            k = cv2.waitKey(0) & 0xFF
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
        
        cv2.destroyAllWindows()
        if endClass: break
    

        allocation=NOTHING
        frName = 'click no animals (ESC to quit, c to continue)'
   #     window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=allocate)
        window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=allocate,onMiddleUpFunction=blockUp,onMiddleDownFunction=blockDown)

        while(1):
            k = cv2.waitKey(0) & 0xFF
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
        cv2.destroyAllWindows() 
        if endClass: break
                

    
            
            
    
    
    
    cap.release()
    
    break
#    fps = round(cap.get(cv2.CAP_PROP_FPS))
#    
#    
#    cap.release()
#    seconds = frCount/(float(fps))
#    
#    startTime=pd.to_datetime(thisDate-datetime.timedelta(seconds=seconds))
#    m, s = divmod(seconds, 60)
#    h, m = divmod(m, 60)
#    duration = datetime.time(int(h), int(m), int(s))
#    # read in the logfile
#    logs = pd.read_csv(logfilename,delim_whitespace=True, header=None)
#    logTimes=pd.to_datetime(logs[0].map(str) + ' ' + logs[1])
#    if thisDate.day==6 and thisDate.month==4:
#        logTimes=logTimes + datetime.timedelta(hours=1)
#        # off by an hour for the first day
#    time_diff = np.array([abs((thing-startTime).total_seconds()) for thing in logTimes])
#    [row_select] = np.where(time_diff<5)
#    
#    
#    print(index, min(time_diff),(lastDate-startTime).total_seconds())
#    if len(row_select)!=1:
#        if len(outDF)==0:
#            angle = 0
#            continue
#        if abs((lastDate-startTime).total_seconds())>3:
#            angle = 0
#            continue
#    else:
#        angle = logs[2].loc[row_select[0]]
#    
#    totalS = totalS + seconds
#    
#    outDF.loc[len(outDF)]=[savedir,thisDate.date(),startTime.time(),duration,thisDate.time(),angle]
#    lastDate = thisDate
#
#
#m, s = divmod(totalS, 60)
#h, m = divmod(m, 60)
#
#print(h,m,s)
#  
#
