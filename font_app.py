import streamlit as st
import numpy as np
import pytesseract
from PIL import Image
from PIL import ImageFilter 

import keras
#import imutils
#from imutils import paths
import os
import sklearn
import tensorflow as tf
from tensorflow import keras

from keras import optimizers
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import imageio
#from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pylab as plt
import matplotlib.image as mpimg

import numpy as np
import PIL
from PIL import ImageFilter
#import cv2
import itertools
import random


# Now you can access the models module
from keras.models import load_model

# Load the model
pretrained_model = load_model('best_model.h5')









@st.cache_data

#Function to load the uploaded image
def load_image(image_file):
    img=Image.open(image_file)
    return img

#Function to generate text from the uploaded image
def image_to_text(image_file):
    text=pytesseract.image_to_string(Image.open(image_file))
    return text

#Preprocessing of input Images 
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image.

    Parameters:
    image_path (str): Path to the input image file.
    target_size (tuple): Target size for resizing the image.

    Returns:
    numpy.ndarray: Preprocessed image as a NumPy array.
    """
    # Open the image file
    image = Image.open(image_path)

    # Resize the image to the target size
    image = image.resize(target_size)

    # Crop the image
    image = crop_image(image)

    # Denoise the image
    image = denoise_image(image)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize the pixel values to the range [0, 1]
    image_array = image_array.astype(np.float32) / 255.0

    return image_array

def crop_image(image, crop_size=(200, 200)):
    """
    Crops an image.

    Parameters:
    image (PIL.Image): Input image.
    crop_size (tuple): Size of the cropped region.

    Returns:
    PIL.Image: Cropped image.
    """
    # Crop the image to the specified size
    width, height = image.size
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def denoise_image(image):
    """
    Denoises an image.

    Parameters:
    image (PIL.Image): Input image.

    Returns:
    PIL.Image: Denoised image.
    """
    # Denoise the image using Gaussian blur
    denoised_image = image.filter(ImageFilter.GaussianBlur(radius=2))

    return denoised_image

st.markdown("<h1 style='text-align: center; color: red;'>Font Recognition App</h1>", unsafe_allow_html=True)
st.subheader('', divider='rainbow')

uploaded_file = st.file_uploader("Upload a image file", 
                                  type=["png","jpg","jpeg"])

if uploaded_file is not None:
    
    
    #file_details={"filename":uploaded_file.name,
                  #"filetype":uploaded_file.type,
                  #"filesize":uploaded_file.size}
    #st.write(file_details)
    # Set image properties
    image_size  = 200  # Pixel width and height
    pixel_depth = 255.0  # Number of levels per pixel
    st.image(load_image(uploaded_file))
    st.markdown("<h2 style='text-align: center; color: black;'>Text Generation from Images </h2>", unsafe_allow_html=True)
    st.subheader('', divider='rainbow')
    st.write(image_to_text(uploaded_file))
    #st.write(preprocess_image(uploaded_file))
    preprocessed_image = preprocess_image(uploaded_file)
    preprocessed_image.shape
    import pandas as pd
    
    
    

    # Load the model
    #pretrained_model = models.load_model('best_model.h5')
     
    
    #from keras.models import load_model
    #pretrained_model= load_model('best_model.keras')
# Make predictions using the pre-trained model
    predictions = pretrained_model.predict(np.expand_dims(preprocessed_image, axis=0))
  
    font_name_dict={0: 'Times_New_Roman',
    1: 'Franklin_Gothic',
    2: 'Arial',
    3: 'Bodoni'}

    # Print the predicted font label
    predicted_font_label = np.argmax(predictions)
    st.markdown("<h2 style='text-align: center; color: black;'>Font Prediction from Text-Images </h2>", unsafe_allow_html=True)
    st.subheader('', divider='rainbow')
    st.write("Predicted Font Label:", predicted_font_label)
    st.write("Predicted Font Label:",font_name_dict[predicted_font_label])
 
   


# Preprocess the image before feeding it into the model
#image_path = 'path_to_your_image.jpg'  # Specify the path to your image
#preprocessed_image = preprocess_image(uploaded_file)

# Load the pre-trained CNN model from a pickle file







