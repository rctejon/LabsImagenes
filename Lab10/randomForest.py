from scipy import io as scipyio
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from skimage import exposure
from tqdm import tqdm

#Carga de informacion
data = scipyio.loadmat('mnist_dataset.mat')

train_data = data['train_data']
train_labels = data['train_labels'][0]
test_data = data['test_data']
test_labels = data['test_labels'][0]

train_data_hogs=[]
test_data_hogs=[]

for image in tqdm(train_data):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(7, 7),
                        cells_per_block=(1, 1), visualize=True)
    train_data_hogs.append(fd)

train_data_hogs = np.array(train_data_hogs)
classifier = RandomForestClassifier(n_estimators=10)

classifier.fit(train_data_hogs, train_labels)
print('Fin Fit')

for image in tqdm(test_data):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(7, 7),
                        cells_per_block=(1, 1), visualize=True)
    test_data_hogs.append(fd)

test_data_hogs = np.array(test_data_hogs)

pred_rf = classifier.predict(test_data_hogs) 
print('Fin Pred')

print(np.vstack((test_labels[0:10],pred_rf[0:10])))
rf_ACA = metrics.accuracy_score(test_labels, pred_rf) # ACA
print(rf_ACA)
