

import cv2
import numpy as np
import time
from deepModels import getSegModel


input_width = 4096
input_height = 2160
modelSeg = getSegModel(input_width, input_height)
filepath='training/weights.hdf5'
modelSeg.load_weights(filepath)
np.set_printoptions(precision=3,suppress=True)
img = cv2.imread('imFull.png')
#img = cv2.imread('yes/img_53.png')
img = cv2.resize(img, (input_width,input_height), cv2.INTER_LINEAR)

X_test = np.array(img, dtype=np.uint8)
X_test = X_test.astype('float32')
X_test = X_test / 255


start_time = time.time()
aa=modelSeg.predict(np.reshape(X_test,(1,input_height,input_width,3)))
print("--- %s seconds ---" % (time.time() - start_time))

#bb=model.predict(np.reshape(X_test[:,:42,],(1,42,42,3)))
#print(bb)

#aaa=np.reshape(aa[:,:,1],(111,157))
aaa=np.reshape(np.argmax(aa,2),(input_height//4,input_width//4)).astype(np.float32)

#print(aaa[11,11])
from matplotlib import pyplot as plt

bb = cv2.resize(aaa,(input_width,input_height), cv2.INTER_LINEAR)
print(bb.shape)
img[bb<0.5,:]=0
# Plot inline
# The important part - Correct BGR to RGB channel
im2 = cv2.cvtColor(115*255.0*aaa, cv2.COLOR_GRAY2RGB)
print(np.max(im2))
# Plot
cv2.imwrite('woohoo.png',img)


