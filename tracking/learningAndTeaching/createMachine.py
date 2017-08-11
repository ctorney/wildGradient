
import glob
import os
import time
import datetime
import numpy as np
import pandas as pd
import cv2
import panZoom
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
DATAHOME = '/home/ctorney/data/tz-2017/'
CODEHOME = '/home/ctorney/workspace/wildGradient/'
inputname = CODEHOME + '/irMovieList.csv'
dfMovies = pd.read_csv(inputname,index_col=0)


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=12),algorithm="SAMME",n_estimators=50)
def reTrain(index):
    #y_pred = 
    clf.fit(outputArray[:counter,0:3],outputArray[:counter,3])
    pickle.dump(clf, open( str(index) + "/classifier.p","wb"))


def is_wildebeest(y,x):#event,x,y,flags,param):
    global counter
 #   if event == cv2.EVENT_LBUTTONDBLCLK:
    print(x,y)
    outputArray[counter,0:3] =  cleanFrame[int(y),int(x),:]
    outputArray[counter,3]=1

        
    counter += 1

def is_zebra(y,x):
    global counter
    outputArray[counter,0:3] =    cleanFrame[int(y),int(x),:]
    outputArray[counter,3]=2
        
    counter += 1
    
def isnt_animal(y,x):
    global counter
    outputArray[counter,0:3] =    cleanFrame[int(y),int(x),:]
    outputArray[counter,3]=0

        
    counter += 1




MAXLEN = 100000

for index,  d in dfMovies.iterrows():

    filename = d['filename']
    
    if not os.path.isdir(str(index)):
        os.mkdir(str(index))
        
    if os.path.isfile(str(index) + "/classifier.p"):
        clf = pickle.load( open( str(index) + "/classifier.p", "rb" ) )
    outputArray = np.zeros((MAXLEN,4))
    counter=0
    if os.path.isfile(str(index) + '/train.npy'):
        oldTrain = np.load(str(index) + '/train.npy')
        counter = len(oldTrain)
        outputArray[:counter,:] = oldTrain[:,:]

    cap = cv2.VideoCapture(filename)
    frCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    endClass = False
    for tt in range(frCount):
    
        _, frame = cap.read()
        cleanFrame = np.copy(frame)
        if frame is None:
            break
        
        if (tt%60)!=0:
            continue
        
    
        if os.path.isfile(str(index) + "/classifier.p"):
            whatIs = clf.predict(cleanFrame.reshape((frame.shape[0]*frame.shape[1],3))).astype(np.uint8)*127
            toggle = cv2.cvtColor(whatIs.reshape((frame.shape[0],frame.shape[1])),cv2.COLOR_GRAY2BGR)
        else:
            toggle = np.copy(frame)
            toggle[:,:,:]=0
        
        swap = np.copy(frame)

        #toggle[:,:,:]=0
    
        cv2.destroyAllWindows()
        frName = 'dbl click wildebeest (ESC to quit, c to continue)'
  #      cv2.namedWindow(frName, flags =  cv2.WINDOW_GUI_EXPANDED )
  #      cv2.setMouseCallback(frName,is_wildebeest)
  #      cv2.imshow(frName,frame)
        window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=is_wildebeest)
        while(1):
            k = cv2.waitKey(0) & 0xFF
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
            elif k==ord('r'):
                reTrain(index)
            elif k==ord('t'):
                swap[:] = toggle[:]
                toggle[:] = frame[:]
                frame[:]=swap[:]
                window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=is_wildebeest)

        cv2.destroyAllWindows()
        if endClass: break
        frName = 'dbl click zebra (ESC to quit, c to continue)'
        window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=is_zebra)
        while(1):
            k = cv2.waitKey(0) & 0xFF
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
            elif k==ord('r'):
                reTrain(index)
            elif k==ord('t'):
                swap[:] = toggle[:]
                toggle[:] = frame[:]
                frame[:]=swap[:]
                window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=is_zebra)
        cv2.destroyAllWindows()
        if endClass: break

        frName = 'dbl click no animals (ESC to quit, c to continue)'
        window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=isnt_animal)
        while(1):
            k = cv2.waitKey(0) & 0xFF
            if k==27:    # Esc key to stop
                endClass = True
                break
            elif k==ord('c'):
                break
            elif k==ord('r'):
                reTrain(index)
            elif k==ord('t'):
                swap[:] = toggle[:]
                toggle[:] = frame[:]
                frame[:]=swap[:]
                window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=isnt_animal)
        cv2.destroyAllWindows()
        if endClass: break
                

    
            
            
    
    
    
    cap.release()
    np.save(str(index) + '/train.npy',outputArray[:counter])
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
