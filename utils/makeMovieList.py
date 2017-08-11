
import glob
import os
import time
import datetime
import numpy as np
import pandas as pd

df = pd.read_csv('gps_log.txt')
dates = df['date'].values.astype(dtype='datetime64')

for filename in glob.iglob('/media/ctorney/My Passport/Tanzania_2017/by-oid/**/*.MP4', recursive=True):
    a = datetime.datetime.fromtimestamp(os.path.getmtime(filename))
    b = np.datetime64(a)
    c = b.astype('datetime64[D]')
    if np.in1d(c,dates)[0]:
        print(filename,',', b )
    