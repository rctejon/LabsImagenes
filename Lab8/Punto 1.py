#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import requests, zipfile
from skimage import io
from sklearn.cluster import KMeans
from skimage.color import rgb2gray

# =============================================================================
#                                   PUNTO 1
# =============================================================================

bins = 100

#Funcion para descargar archivos desde google drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#Descarga del .zip desde google drive y posterior extraccion de las imagenes
file_name = 'dogs.zip'
download_file_from_google_drive('1Tb6DsgVJhHmsqLA-e1xYuphpSM-zCvjI', file_name)
with zipfile.ZipFile(file_name, 'r') as z:
    z.extractall()
 
#Obtener los paths de las imagenes de los dalmatas
files = [f for f in glob.glob(os.path.join('dogs','dalmatian','*.jpg'), recursive=True)]
def get_number(x):
    arr = x.split('\\')
    return int(arr[2].replace('.jpg',''))
files = sorted(files, key=get_number)
dalmatians = [(io.imread(os.path.join(f))) for f in files]

#Obtener los paths de las imagenes de los dalmatas
files = [f for f in glob.glob(os.path.join('dogs','golden','*.jpg'), recursive=True)]
files = sorted(files, key=get_number)
goldens = [(io.imread(os.path.join(f))) for f in files]

#Calcular los histogramas de cada una de las imagenes
dalmatians_histograms = [np.histogram(rgb2gray(image).ravel(), bins, [0,1])[0] for image in dalmatians]
goldens_histograms = [np.histogram(rgb2gray(image).ravel(), bins, [0,1])[0] for image in goldens]

#Extraer el set de datos de entrenamiento
training_set = np.asarray(dalmatians_histograms[1::2] + goldens_histograms[1::2])
training_set_images = np.asarray(dalmatians[1::2] + goldens[1::2])

#Extraer el set de datos de prueba
test_set = np.asarray(dalmatians_histograms[::2] + goldens_histograms[::2])
test_set_images = np.asarray(dalmatians[::2] + goldens[::2])

#Semilla del generador para reproducibilidad
seed = 1313 
np.random.seed(seed)

#Aplicar k means con 2 clusters sobre los histogramas
k = 2
kmeans_model = KMeans(n_clusters=k, random_state=seed)
trained_model = kmeans_model.fit(training_set)

#Mostrar las clasificaciones de las imagenes de entrenamiento
labels = trained_model.labels_.astype(np.uint8)

images_0 = training_set_images[labels==0]
images_1 = training_set_images[labels==1]

plt.suptitle(f'Bins = {bins}')
width = max(len(images_0), len(images_1))

index = 1
for i in range(len(images_0)):
    plt.subplot(4, width, index)
    plt.imshow(images_0[i])
    plt.title("Label 0")
    plt.axis('off')
    index = index+1
    
index = width + 1    
for i in range(len(images_0)):
    plt.subplot(4, width, index)
    plt.hist(rgb2gray(images_0[i]).ravel(), bins=bins)
    index = index+1
    plt.axis('on')
    
index = 2*width + 1
for i in range(len(images_1)):
    plt.subplot(4, width, index)
    plt.imshow(images_1[i])
    index = index+1
    plt.title("Label 1")
    plt.axis('off')
    
index = 3*width + 1   
for i in range(len(images_1)):
    plt.subplot(4, width, index)
    plt.hist(rgb2gray(images_1[i]).ravel(), bins=bins)
    index = index+1
    plt.axis('on')
    
plt.tight_layout()
plt.show()
   
#Clasificar las imagenes segun el rtado 
labels = kmeans_model.predict(test_set)

images_0 = test_set_images[labels==0]
images_1 = test_set_images[labels==1]

plt.suptitle(f'Bins = {bins}')
width = max(len(images_0), len(images_1))

index = 1
for i in range(len(images_0)):
    plt.subplot(4, width, index)
    plt.imshow(images_0[i])
    plt.title("Label 0")
    plt.axis('off')
    index = index+1
    
index = width + 1    
for i in range(len(images_0)):
    plt.subplot(4, width, index)
    plt.hist(rgb2gray(images_0[i]).ravel(), bins=bins)
    index = index+1
    plt.axis('on')
    
index = 2*width + 1
for i in range(len(images_1)):
    plt.subplot(4, width, index)
    plt.imshow(images_1[i])
    index = index+1
    plt.title("Label 1")
    plt.axis('off')
    
index = 3*width + 1   
for i in range(len(images_1)):
    plt.subplot(4, width, index)
    plt.hist(rgb2gray(images_1[i]).ravel(), bins=bins)
    index = index+1
    plt.axis('on')
    
plt.tight_layout()
plt.show()