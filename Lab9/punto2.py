from scipy import io as scipyio
from scipy import signal
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
import os
from tqdm import tqdm
import random
import math

mat = scipyio.loadmat('filterbank.mat')
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def CalculateFilterResponse_201424311_201617853(grayscale_image, filters):
    resp=[]
    print(grayscale_image.shape)
    for filt in filters:
        imFil=signal.correlate2d(grayscale_image,filt, boundary='symm', mode='same')
        resp.append(imFil)
    resp = np.array(resp)
    print(resp.shape)
    res = []
    for i in range(len(grayscale_image)):
        for j in range(len(grayscale_image[0])):
            res.append(resp[:,i,j])
    return res

def CalculateTextonDictionary_201424311_201617853(image_names, filters, k,filename):
    # if len(scipyio.loadmat('textonDictionary_201424311_201617853.mat')['model'])>0:
    #     return scipyio.loadmat('textonDictionary_201424311_201617853.mat')['model']
    vectorArray=[]
    for im in tqdm(image_names):
        image = rgb2gray(mpimg.imread(os.path.join(im)))
        vectorArray.append(CalculateFilterResponse_201424311_201617853(image,filters))
    vectorArray = np.concatenate(vectorArray, axis=0)
    model = KMeans(n_clusters=k, random_state=0).fit(vectorArray)
    scipyio.savemat(filename, {'model': model.cluster_centers_})
    return model.cluster_centers_

def CalculateTextonHistogram_201424311_201617853(grayscale_image, centroids,dist):
    hist = [0]*len(centroids)
    mat = scipyio.loadmat('filterbank.mat')
    filterResponse = CalculateFilterResponse_201424311_201617853(grayscale_image,mat['filterbank'])
    print(len(filterResponse))
    for res in filterResponse:
        dists = np.array([dist(res,centroid) for centroid in centroids])
        centroidIndex =  np.argmin(dists)
        hist[centroidIndex]+=(1/len(filterResponse))
    print(hist)
    return hist

def PredictClass_201424311_201617853(grayscale_image, train_hists, dist):
    listImage = [m[0:-4]  for m in os.listdir('train')]
    res = CalculateTextonHistogram_201424311_201617853(grayscale_image,scipyio.loadmat('textonDictionary_201424311_201617853.mat')['model'],euclidianDist)
    dists = np.array([dist(res,hist) for hist in train_hists])
    resArg = np.argmin(dists)
    print(listImage[resArg])
    return listImage[resArg]
    

def euclidianDist(X, Y):
    d = 0
    for i in range(len(X)):
        d += (X[i]-Y[i])**2
    return d

def intersectionDist(X, Y):
    d = 0
    for i in range(len(X)):
        d += min(X[i],Y[i])
    return d


mat = scipyio.loadmat('filterbank.mat')

listImage = [os.path.join('train',m)  for m in os.listdir('train')]

centroids = CalculateTextonDictionary_201424311_201617853(listImage,mat['filterbank'],2,'textonDictionaryk=3.mat')
print(centroids)
train_hists=[]
for im in tqdm(listImage):
    image = rgb2gray(mpimg.imread(os.path.join(im)))
    train_hists.append(CalculateTextonHistogram_201424311_201617853(image,centroids,euclidianDist))

print(train_hists)

listTest = [os.path.join('test',m)  for m in os.listdir('test')]
preds=[]
print(listTest)
for im in tqdm(listTest):
    image = rgb2gray(mpimg.imread(os.path.join(im)))
    preds.append(PredictClass_201424311_201617853(image,train_hists,intersectionDist))
print(preds)
