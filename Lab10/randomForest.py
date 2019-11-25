from scipy import io as scipyio
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from skimage import exposure
from tqdm import tqdm
from sklearn.externals import joblib


#Carga de informacion
data = scipyio.loadmat('mnist_dataset.mat')
filename = 'random_forest3.sav'
numArboles=50
alturaArbol=None
orienta=8
cellSize=(4,4)

loaded_model=None
classifier=None
try:
    loaded_model = joblib.load(filename)
except FileNotFoundError:
    pass

train_data = data['train_data']
train_labels = data['train_labels'][0]
test_data = data['test_data']
test_labels = data['test_labels'][0]

train_data_hogs=[]
test_data_hogs=[]

if loaded_model==None:
    for image in tqdm(train_data):
        fd, hog_image = hog(image, orientations=orienta, pixels_per_cell=cellSize,
                            cells_per_block=(1, 1), visualize=True)
        train_data_hogs.append(fd)

    train_data_hogs = np.array(train_data_hogs)
    classifier = RandomForestClassifier(n_estimators=numArboles,max_depth=alturaArbol)

    classifier.fit(train_data_hogs, train_labels)
    print('Fin Fit')

if classifier!=None:
    joblib.dump(classifier, filename)

for image in tqdm(test_data):
    fd, hog_image = hog(image, orientations=orienta, pixels_per_cell=cellSize,
                        cells_per_block=(1, 1), visualize=True)
    test_data_hogs.append(fd)

test_data_hogs = np.array(test_data_hogs)


loaded_model = joblib.load(filename)
pred_rf = loaded_model.predict(test_data_hogs) 
print('Fin Pred')

print(np.vstack((test_labels[0:10],pred_rf[0:10])))
rf_ACA = metrics.accuracy_score(test_labels, pred_rf) # ACA
print(rf_ACA)
