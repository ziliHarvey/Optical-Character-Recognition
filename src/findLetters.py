import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    #denoise 
    image = skimage.restoration.denoise_wavelet(image,multichannel=True, convert2ycbcr=True)
    #greyscale
    image = skimage.filters.gaussian(image,3.2)
    image = skimage.color.rgb2gray(image)
    
    
    #threshold
    image = skimage.morphology.erosion(image)
    threshold = skimage.filters.threshold_otsu(image)
    #morphology: The morphological closing on an image is defined as a dilation followed by an erosion. 
    #Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks. 
    bw = skimage.morphology.closing(image>threshold,skimage.morphology.square(3))
    
    
    bw = (bw == False)
    #remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)
    label_image = skimage.measure.label(cleared)
#    image_label_overlay = skimage.color.label2rgb(label_image, image=image)
    #skip small boxes
    for region in skimage.measure.regionprops(label_image):
        if region.area > 500:
            minr, minc, maxr, maxc = region.bbox
            minr -= 3
            minc -= 3
            maxr += 3
            maxc += 3
            bboxes.append((minr, minc, maxr, maxc))
    bw = (bw == False)
    return bboxes, bw