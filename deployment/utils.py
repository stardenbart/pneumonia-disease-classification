import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import requests
from io import BytesIO
import numpy as np
from numpy import reshape
import tensorflow as tf
from PIL import Image


def preprocess(image):

    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0

    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array[0]

    return image_array

def download_from_dropbox(dropbox_url, output_path):

    # Modify the Dropbox URL for direct download
    direct_url = dropbox_url.replace('?dl=0', '?dl=1')
    response = requests.get(direct_url)
    with open(output_path, 'wb') as file:
        file.write(response.content)

# model-pneumonia-vgg19.h5
def modelV1():
    dropbox_url = 'https://www.dropbox.com/scl/fi/j2o84a49h9cln02c34yyx/model-pneumonia-vgg19.h5?rlkey=tvyskouimeec92axk9wfx88w6&st=pahmxhat&dl=1'
    output_path = 'modelv1.h5'
    download_from_dropbox(dropbox_url, output_path)

    loaded_model = load_model('modelv1.h5')

    return loaded_model

# pneumoniamodelvgg19.h5
def modelV2():
    

    dropbox_url = 'https://www.dropbox.com/scl/fi/jyegyy1nomvlgqpk2cqmb/pneumoniamodelvgg19.h5?rlkey=rgl8p1pnhm2s6mlz06h3uvswq&st=tcx72hr2&dl=1'
    output_path = 'modelv2.h5'
    download_from_dropbox(dropbox_url, output_path)

    loaded_model = load_model('modelv2.h5')

    return loaded_model


def binary_predict_image(model, image, threshold=0.5):
    # Load dan preprocess gambar
    img_array = preprocess(image)
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    
    # Atur confidence
    confidence = prediction[0]
    if 0.35 <= confidence <= 0.5 or 0.5 < confidence <= 0.65:
        confidence_category = 'Low Confidence'
    elif 0.20 <= confidence < 0.35 or 0.65 < confidence <= 0.80:
        confidence_category = 'Medium Confidence'
    else:
        confidence_category = 'High Confidence'
    
    # Tentukan kelas prediksi berdasarkan threshold
    predicted_class_label = 'Pneumonia' if prediction[0] > threshold else 'Normal'
    
    return predicted_class_label, confidence_category

label_array = ['PNEUMONIA','NORMAL']