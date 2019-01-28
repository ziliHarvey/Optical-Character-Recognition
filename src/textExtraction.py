import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from findLetters import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.figure()

    plt.imshow(bw, cmap=plt.cm.gray)
    plt.imshow(im1)
    #skimage.io.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    
    
import pickle
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))

num = 0
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    num += 1
    minrList = []
    result = ''
    #sort bboxes to make sure it is selected left to right, top to bottom as in image
    start = 0
    count = 0
    bboxes_new = []
    for i in range(len(bboxes)):
        count += 1
        if abs(bboxes[i][0] - bboxes[i+1][0]) > 100:
            bboxes_s = bboxes[start:count]
            bboxes_s.sort(key = lambda x:x[1])
            bboxes_new.extend(bboxes_s)
            start = count
        if i == len(bboxes)-2:
            bboxes_s = bboxes[start:]
            bboxes_s.sort(key = lambda x:x[1])
            bboxes_new.extend(bboxes_s)
            break
        
    for bbox in bboxes_new:
        minr, minc, maxr, maxc = bbox
        minrList.append(minr)
        crop = (bw[minr:(maxr+1),minc:(maxc+1)]).astype(float)
        #pad zero
        crop = np.pad(crop,(int((maxr-minr)/5),int((maxc-minc)/5)),'edge')
        crop = skimage.transform.resize(crop,(32,32))
        crop_T = crop.T
        xb = crop_T.ravel().reshape(1,1024)
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        result += letters[np.argmax(probs)]
        #pack words according to rows
    row = 1
    count = 0
    start = 0
    for i in range(len(minrList)):
        count += 1
        if i == len(minrList) -1:
            break
        if abs(minrList[i] - minrList[i + 1]) > 100:
            print(result[start:count])
            start = count
            row += 1
    print(result[start:count])
    print('there are %d rows in %d-th image'%(row,num))
    
    
                
    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
