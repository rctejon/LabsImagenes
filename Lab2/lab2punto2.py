import os
import requests
import numpy as np
import  nibabel as nib
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

file_start='Brats17_2013_10_1_'
file_ends=['flair','t1','t1ce','t2','seg']
mris=[]

for end in file_ends:
    mri=nib.load(os.path.join('data',file_start+end,file_start+end+'.nii'))
    mris.append(mri)

size=(240,240,155,4)
vol = np.zeros(size,dtype=np.single)

for i in range(len(mris)-1):
    vol[:,:,:,i]=mris[i].get_fdata()
    

print(vol.shape)