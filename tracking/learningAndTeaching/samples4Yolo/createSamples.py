
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




def allocate(y,x):#event,x,y,flags,param):
    print("middle button to create")


def blockStop(y,x):
    startX = int(min(x,blockSelectX))
    stopX = int(max(x,blockSelectX))
    
    startY = int(min(y,blockSelectY))
    stopY = int(max(y,blockSelectY))
    

def blockStart(y,x):
    global blockSelectX
    global blockSelectY
    blockSelectX=x
    blockSelectY=y
    

blockSelectX=0
blockSelectY=0


imName = 'test.png'

outFile = os.path.splitext(os.path.basename(imName))[0] + '.txt'
output = open(outFile, "w")
output.write("Purchase Amount: %s\n" % 1)
output.write("Purchase Amount: %s\n" % 1)
output.close()
frName = 'click wildebeest (ESC to quit, c to continue)'
frame = cv2.imread(imName)
window = panZoom.PanZoomWindow(frame, frName, onLeftClickFunction=allocate,onMiddleUpFunction=blockStop,onMiddleDownFunction=blockStart)
while(1):
    k = cv2.waitKey(0) & 0xFF
    if k==27:    # Esc key to stop
        break
cv2.destroyAllWindows()
    
    



