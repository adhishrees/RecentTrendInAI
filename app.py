import streamlit as st
import pickle
import numpy as np
import requests
from io import BytesIO

# Google Drive direct download link (without /view?usp=sharing)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1L6JLG0paJ-Od7PH-YQSsxIJPjYg5X5Lr"

@st.cache_resource  # cache so it only downloads once
def load_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    return pickle.load(BytesIO(response.content))

# Load the model
best_model = load_model()

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Streamlit app
st.title("Fashion-MNIST Classifier")

st.header("Upload Your Image")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    from PIL import Image

    # Read and preprocess the image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = image.resize((28, 28))  # resize
    image_array = np.array(image)
    image_array = 255 - image_array  # invert colors
    image_array = image_array / 255.0  # normalize
    image_flat = image_array.reshape(1, -1)  # flatten

    # Display original uploaded image
    st.image(uploaded_file, caption="Original Uploaded Image", width=200)

    # Predict
    prediction = best_model.predict(image_flat)
    predicted_class = class_names[prediction[0]]

    st.subheader("Prediction")
    st.write(f"The uploaded image is classified as: **{predicted_class}**")
