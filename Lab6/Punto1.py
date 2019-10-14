#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests, zipfile
import scipy.signal as sc
import scipy
import pdb
from skimage import io
from skimage.color import rgb2lab, rgb2hsv, lab2rgb, hsv2rgb

# ------------
# --PUNTO 1---
# -----------------------------------------------------
# =============================================================================
#                                   PUNTO 1
# =============================================================================

# =============================================================================
# Function that converts from one color space to another and then manipulates a
# certain channel
# * image - an RGB image
# * dest_space - the wanted color space: {’rgb’,’hsv’,’lab’}
# * channel - a channel from the chosen color space
# * value - 0 lowest possible value for the channel
#           1 average between lowest and higest possible value
#           2 highest possible value
# =============================================================================
def MyModifyChannel_201424311_201617853(image, dest_space, channel, value):
    #First map image from RGB to the desired color space
    if dest_space == 'rgb':
        new_image = image.copy()
    elif dest_space == 'hsv':
        new_image = rgb2hsv(image)
    elif dest_space == 'lab':
        new_image = rgb2lab(image)
    else:
        raise Exception(f'{dest_space} is not a valid color space. Use one of the following {{\'rgb\',\'hsv\',\'lab\'}}')
    
    #Modifies channel according to value
    if channel == dest_space[0]:
        minimum = np.min(new_image[:,:,0])
        maximum = np.max(new_image[:,:,0])
        if value == 0:
            new_image[:,:,0] = minimum * np.ones(new_image[:,:,0].shape)
        elif value == 1:
            new_image[:,:,0] = (maximum+minimum)/2 * np.ones(new_image[:,:,0].shape)
        elif value == 2:
            new_image[:,:,0] = maximum * np.ones(new_image[:,:,0].shape)
        else:
            raise Exception(f'value = {value} is not a valid parameter. Please use a value from 0 to 2')
    elif channel == dest_space[1]:
        minimum = np.min(new_image[:,:,1])
        maximum = np.max(new_image[:,:,1])
        if value == 0:
            new_image[:,:,1] = minimum * np.ones(new_image[:,:,1].shape)
        elif value == 1:
            new_image[:,:,1] = (maximum+minimum)/2 * np.ones(new_image[:,:,1].shape)
        elif value == 2:
            new_image[:,:,1] = maximum * np.ones(new_image[:,:,1].shape)
        else:
            raise Exception(f'value = {value} is not a valid parameter. Please use a value from 0 to 2')
    elif channel == dest_space[2]:
        minimum = np.min(new_image[:,:,2])
        maximum = np.max(new_image[:,:,2])
        if value == 0:
            new_image[:,:,2] = minimum * np.ones(new_image[:,:,2].shape)
        elif value == 1:
            new_image[:,:,2] = (maximum+minimum)/2 * np.ones(new_image[:,:,2].shape)
        elif value == 2:
            new_image[:,:,2] = maximum * np.ones(new_image[:,:,2].shape)
        else:
            raise Exception(f'value = {value} is not a valid parameter. Please use a value from 0 to 2')
    else:
        raise Exception(f'{channel} is not a valid channel for the {dest_space} color space selected')
        
    #Map the result from the last step to RGB space
    if dest_space == 'rgb':
        new_image = new_image
    elif dest_space == 'hsv':
        new_image = hsv2rgb(new_image)
    elif dest_space == 'lab':
        new_image = lab2rgb(new_image)
      
    return new_image

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
file_name = 'colorful_ims.zip'
download_file_from_google_drive('1qGzCj0Fo4cvK9v97zec05t9GZe6-OIg9', file_name)
with zipfile.ZipFile(file_name, 'r') as z:
    z.extractall()
 
#Obtener los paths de cada una de las imagenes
path = os.path
files = [f for f in glob.glob(os.path.join('colorful_ims','*.jpg'), recursive=True)]

#Obtener un array con todas las imagenes 
images = [io.imread(os.path.join(f)) for f in files]

commands = [['hsv', 's', 2],
            ['lab', 'b', 0],
            ['hsv', 'h', 0],
            ['rgb', 'b', 0]]

for i in range(len(images)):
    image = images[i]
    command = commands[i]
    plt.suptitle(f'Image {i+1}')
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(image)
    new_image = MyModifyChannel_201424311_201617853(image, command[0], command[1], command[2])
    plt.subplot(1, 2, 2)
    plt.title('Modified')
    plt.imshow(new_image)
    plt.show()
    input("Press Enter to continue...")