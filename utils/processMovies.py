
import glob
import os
import time
import datetime
import numpy as np
import pandas as pd
import cv2
import shutil


inputname = '../movieList.csv'
outDF = pd.read_csv(inputname,index_col=0)

totSize = 0
for index,  d in outDF.iterrows():

    filename = d['filename']
    savename = os.path.basename(filename)
    savedir = '/home/ctorney/data/tz-2017/ir/' + savename
    if os.path.isfile(savedir):
        savedir = '/home/ctorney/data/tz-2017/vis/' + savename
        shutil.copy(filename, savedir)
    else:
        shutil.copy(filename, savedir)
    totSize = totSize + (os.path.getsize(filename)/1000000000)


