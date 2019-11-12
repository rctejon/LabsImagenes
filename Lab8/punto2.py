import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
import os
import imageio
import threading


# =============================================================================
#                                   PUNTO 2
# =============================================================================

def MyVQ_training_201424311_201617853(grayscale_image, window_size, k):
    num1=int((window_size-1)/2)
    vectorArray = []
    labeled_image = np.zeros_like(grayscale_image)
    for i in range(num1, len(grayscale_image)-num1):
        for j in range(num1, len(grayscale_image[i])-num1):
            window = grayscale_image[i-num1:i+num1+1,j-num1:j+num1+1]
            vector = window.reshape((1,window_size**2))
            vectorArray.append(vector)
    vectorArray = np.concatenate(vectorArray,axis=0)
    model = KMeans(n_clusters=k, random_state=0).fit(vectorArray)
    print(model)
    num = 0
    for i in range(num1, len(grayscale_image)-num1):
        for j in range(num1, len(grayscale_image[i])-num1):
            labeled_image[i,j] = model.labels_[num]
            num += 1
    return model , labeled_image

def MyVQ_predict_201424311_201617853(grayscale_image, window_size, model):
    num1=int((window_size-1)/2)
    labeled_image = np.zeros_like(grayscale_image)
    for i in range(num1, len(grayscale_image)-num1):
        for j in range(num1, len(grayscale_image[i])-num1):
            window = grayscale_image[i-num1:i+num1+1,j-num1:j+num1+1]
            vector = window.reshape((1,window_size**2))
            labeled_image[i,j] = model.predict(vector)
    return labeled_image

images = []
dirs = os.listdir('video')
k = 0
for im in dirs:
    if k%7==0:
        images.append(rgb2gray(io.imread(os.path.join('video',im))))
    k+=1

model, seg = MyVQ_training_201424311_201617853(images[0], 3,2)

gifImages = images.copy()

def segImage( i ):
    global images, gifImages

with imageio.get_writer(os.path.join('duck1.gif'), mode='I') as writer:
    for i in range(len(images)):
        print('inicio ',i)
        gifImages[i] = MyVQ_predict_201424311_201617853(images[i],3,model) 
        print('fin',i)
        writer.append_data(gifImages[i])