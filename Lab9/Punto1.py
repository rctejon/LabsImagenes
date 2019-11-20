#Librerias utilizadas
import numpy as np
import os, glob
import requests, zipfile
from skimage import io
from sklearn.metrics import accuracy_score

# No mostar las advertencia si se divide por 0 o se aca log(0)
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
#                  FUNCIONES UTILIZADAS
# =============================================================================

# =============================================================================
# Funcion que calcula el histograma conjunto de color de una imagen
# * color_im - imagen a color (RGB)
# * bins - numero de bins a considerar por cada canal
# El histograma retornado esta normalizado de acuerdo a la norma L1
# =============================================================================
def ComputeColorHist_201424311_201617853(color_im, bins):
    size = 256./bins # Tamano de cada bin
    bin_edges = np.linspace(0, 256, bins+1) # Limites de cada bin
    bin_centers = np.linspace(0 + size/2, 256-size/2, bins) # Valor central de los bins
    # Construccion de lista de tuplas que representan el valor central de cada 
    # uno de los bins
    bins_ = []
    for i in bin_centers:
        for j in bin_centers:
            for k in bin_centers:
                bins_.append((i, j, k))
    # Construccion del histograma de color
    histogram = np.zeros(bins*bins*bins)
    for i in range(color_im.shape[0]):
        for j in range(color_im.shape[1]):
            #Extraccion de las intensidades del pixel en la posicion i,j
            pixel = color_im[i][j][:] 
            # Determinar el bin de acuerdo a las intensidades en cada canal
            c0 = min(int(pixel[0]/size), bins-1)
            c1 = min(int(pixel[1]/size), bins-1)
            c2 = min(int(pixel[2]/size), bins-1)
            #Aumentar el contador del bin en 1
            histogram[c0*bins*bins + c1*bins + c2] += 1
    return (bins_, histogram)

# =============================================================================
# Funcion que calcula la distancia entre el histograma 1 y el 2 segun el 
# criterio de distancia seleccionado
# * hist1 - Histograma 1
# * hist2 - Histograma 2
# * crit - string que define el criterio para calcular la distancia.
#          Puede ser : 'intersect', 'chi' o 'kl'
# =============================================================================
def compute_distance(hist1, hist2, crit):
    if (crit == 'intersect'):
        d = np.minimum(hist1, hist2)
    elif (crit == 'chi'):
        d = np.divide((hist1-hist2)**2, hist1+hist2)
    elif (crit == 'kl'):
        d = np.multiply(hist2-hist1, np.log(np.divide(hist2, hist1)))  
    else:
        raise Exception(f'{crit} no es un criterio valido. Seleccionar uno de'+ 
                        'los siguientes: [intersect, chi o kl]')
    return np.nansum(d)

# =============================================================================
# Funcion que calcula todas las medidas de diferencia (o similitud) entre el 
# histograma hist_test y los histogramas en hists_train_dataset  y dadas estas 
# medidas determina cual es el histograma mas parecido a hist_test.
# Retorna la etiqueta correspondiente a dicho histograma.
# * hist_test - Histograma (np.array de longitud b) que corresponde a un histo-
#   grama conjunto de color de una imagen.
# * hists_train_dataset - Matriz de dimensiones n x b. Cada fila de la matriz
#   corresponde a un histograma de color conjunto de una imagen del conjunto
#   de datos de entrenamiento.
# * labels_train_dataset - Lista o vector de longitud n, cuya i-esima entrada
#   indica la etiqueta (clase) del i-esimo histograma de hists_train_dtaaset.
#   Es decir, la clase de la i-esima imagen.
# * crit - String que define el criterio de diferencia (o similitud) para cal-
#   cular la distancia entre dos histogramas. Es 'intersect', 'chi' o 'kl'
# =============================================================================                
def ComputeHistNN_2014124311_201617853(hist_test, hists_train_dataset, 
                                       labels_train_dataset, crit):
    # Dimensiones de las entradas
    n = hists_train_dataset.shape[0]
    b = hists_train_dataset.shape[1]
    n2 = hist_test.shape[0]
    b2 = len(labels_train_dataset)
    # Verificacion de que hist_test es un vector
    # Verificacion de que las dimensiones de las entradas coincidan
    if(b != b2):
        raise Exception(f'La matriz hists_train_datasets debe tener dimension'+
                        f'{n2} x {b2} pero es de dimension {n} x {b}.')
    elif(n != n2):
        raise Exception(f'Lista de etiquetas debe tener longiutd {n}, no {n2}')
    # Determinar si la eleccion del mejor histograma se hace maximizando  o 
    # minimizando la distancia.
    maximize = (crit == 'intersect')
    # Inicializacion de variables para guardar el histograma mas parecido    
    best = 0 if maximize else np.Inf
    best_index = -1
    for i in range(n):
        # Calcular la distancia entre hist_test y hist_train
        hist_train = hists_train_dataset[i]
        distance = compute_distance(hist_test, hist_train, crit)
        # Determina si hist_train es el histograma mas parecido a hist_test 
        # (de momento). Si crit es 'intersect' lo hace escogiendo el maximo.
        best = max(best, distance) if maximize else max(best, distance)
        best_index = i if best == distance else best_index
    return labels_train_dataset[best_index]

# =============================================================================
# Algoritmo de clasificacion de imagenes basado en el vecino mas cercano.
# * ims_train - Imagenes del set de entrenamiento.
# * ims_test - Imagenes del set de prueba.
# * labels_train - Lista o vector de longitud n, cuya i-esima entrada
#   indica la etiqueta (clase) del i-esimo histograma de hists_train_dtaaset.
#   Es decir, la clase de la i-esima imagen.
# * bins - Numero de bins para calcular el histograma conjunto de color.
# * crit - String que define el criterio de diferencia (o similitud) para cal-
#   cular la distancia entre dos histogramas. Valores:'intersect', 'chi' o 'kl'
# =============================================================================                
def classification_algorithm(ims_train, ims_test, labels_train, bins, crit):
    n = len(ims_train) # Numero de elementos del set de entrenamiento
    # Calcular para todas las imagenes de entrenamiento el descriptor de histo-
    # grama de color conjunto
    hists_train = np.zeros((n, bins**3))
    for (i, image) in enumerate(ims_train):
        hists_train[i] = ComputeColorHist_201424311_201617853(image, bins)[1]
    # Calcular para cada imagen de test la etiqueta del vecino más cercano, con 
    # base en el criterio crit. Este paso constituye la generación las predic-
    # ciones del algoritmo.
    labels_pred = []
    for (i, image) in enumerate(ims_test):
        hist_test = ComputeColorHist_201424311_201617853(image, bins)[1]
        label = ComputeHistNN_2014124311_201617853(hist_test, hists_train, 
                                                   labels_train, crit)
        labels_pred.append(label)
    # Retornar las predicciones de clase segun el algoritmo.
    return labels_pred
   
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

# Parametros usados
n_bins = [4, 8]
crits = ['intersect', 'chi', 'kl']

# Descarga del .zip desde google drive y posterior extraccion de las imagenes
file_name = 'data.zip'
download_file_from_google_drive('1YSLlYP9La4YlQA9b6S_VLoHKOr-8S-qV', file_name)
with zipfile.ZipFile(file_name, 'r') as z:
    z.extractall()
 
# Obtener los paths, imagenes y labels de las imagenes del set de entrenamiento
files = [f for f in glob.glob(os.path.join('data','train','*.jpg'))]
files = files + [f for f in glob.glob(os.path.join('data','train','*.jpeg'))]
ims_train = [(io.imread(os.path.join(f))) for f in files]
labels_train = [os.path.basename(file).split('.')[0][:-1] for file in files]

# Obtener los paths, imagens y labels de las imagenes del set de prueba
files = [f for f in glob.glob(os.path.join('data','test','*.jpg'))]
files = files + [f for f in glob.glob(os.path.join('data','test','*.jpeg'))]
ims_test = [(io.imread(os.path.join(f))) for f in files]
labels_test = [os.path.basename(file).split('.')[0][:-1] for file in files]

# Correr el algoritmo de clasificacion para cada una de las imagenes del set de 
# prueba variando los parametros
for bins in n_bins:
    for crit in crits:
        labels_pred = classification_algorithm(ims_train, ims_test, 
                                               labels_train, bins, crit)
        aca = accuracy_score(labels_test, labels_pred)
        print(f'Para bins = {bins} y crit = {crit} se obtuvo un ACA de {aca}')