import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import cv2
import numpy as np

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)



def classify(image, model):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class 
    and confidence score of the image.

    Parameters:
        image (bytes): The image data as bytes.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Decode the image from bytes
    img_array = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    # Resize and preprocess the image
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    label_mapping = {
    0: 'Malignant',
    1: 'Benign'
     }
    predicted_class_name = label_mapping[predicted_class_index]

    return predicted_class_name