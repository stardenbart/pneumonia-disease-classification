import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19
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

def modelV1():

    model_path= '..\model\pneumoniamodelvgg19.h5'
    loaded_model = tf.keras.models.load_model(model_path)

    return loaded_model

def modelV2():

    model_path= '..\model\model-pneumonia-vgg19.h5'
    loaded_model = tf.keras.models.load_model(model_path)

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