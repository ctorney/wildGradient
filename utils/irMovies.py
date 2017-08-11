
import glob
import os
import time
import datetime
import numpy as np
import pandas as pd
import cv2

HOME = '/home/ctorney/data/tz-2017/'
inputname = '../movieList.csv'
df = pd.read_csv(inputname,index_col=0)

outputname = '../irMovieList.csv'
columns = ['filename','date','starttime','duration','stoptime','angle']

outDF = pd.DataFrame(columns=columns)

totalS = 0

totSize = 0
lastDate=pd.to_datetime(0)
for index,  d in df.iterrows():

    filename = d['filename']
    savename = os.path.basename(filename)
    savedir = HOME + '/ir/' + savename
    if not os.path.isfile(filename):
        continue
    thisDate=pd.to_datetime(d['date'])
    

    logfilename = HOME + '/logs/' + thisDate.strftime('%Y%m%d') + '.txt'

    if not os.path.isfile(logfilename):
        continue
    
    
    
    
    cap = cv2.VideoCapture(filename)
    frCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    
    cap.release()
    seconds = frCount/(float(fps))
    
    startTime=pd.to_datetime(thisDate-datetime.timedelta(seconds=seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    duration = datetime.time(int(h), int(m), int(s))
    # read in the logfile
    logs = pd.read_csv(logfilename,delim_whitespace=True, header=None)
    logTimes=pd.to_datetime(logs[0].map(str) + ' ' + logs[1])
    if thisDate.day==6 and thisDate.month==4:
        logTimes=logTimes + datetime.timedelta(hours=1)
        # off by an hour for the first day
    time_diff = np.array([abs((thing-startTime).total_seconds()) for thing in logTimes])
    [row_select] = np.where(time_diff<5)
    
    
    print(index, min(time_diff),(lastDate-startTime).total_seconds())
    if len(row_select)!=1:
        if len(outDF)==0:
            angle = 0
            continue
        if abs((lastDate-startTime).total_seconds())>3:
            angle = 0
            continue
    else:
        angle = logs[2].loc[row_select[0]]
    
    totalS = totalS + seconds
    
    outDF.loc[len(outDF)]=[savedir,thisDate.date(),startTime.time(),duration,thisDate.time(),angle]
    lastDate = thisDate


m, s = divmod(totalS, 60)
h, m = divmod(m, 60)

print(h,m,s)
  
outDF.to_csv(outputname)

