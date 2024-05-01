import base64
import streamlit as st
from keras.models import load_model
from PIL import Image

from util import classify

# Function to set background
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

# Set background
set_background('/Users/hajaryahia/Downloads/breast-cancer/bgs/bg5.jpg')

# Set title
st.title('Breast Cancer classification')

# Set header
st.header('Please upload a Breast Mammography image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png', 'pgm'])

# Load classifier
model = load_model('/Users/hajaryahia/Downloads/model-final.h5')

# Display image and classify
if file is not None:
    image = file.read()  # Read the uploaded image file as bytes
    
    # Classify image
    class_name = classify(image, model)

    # Display the uploaded image
    img = Image.open(file).convert('RGB')
    st.image(img, use_column_width=True)

    # Write classification result
    st.write("## Predicted Class: {}".format(class_name))
