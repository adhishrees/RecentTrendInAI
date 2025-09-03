import streamlit as st
import pickle
import numpy as np

# Load the best model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Streamlit app
st.title("Fashion-MNIST Classifier")

st.header("Upload Your Image")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    from PIL import Image
    import cv2

    # Read and preprocess the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors
    image_array = image_array / 255.0  # Normalize
    image_flat = image_array.reshape(1, -1)  # Flatten

    # Display the original uploaded image before any transformations
    st.image(uploaded_file, caption="Original Uploaded Image", width=200)

    # Predict the class
    prediction = best_model.predict(image_flat)
    predicted_class = class_names[prediction[0]]

    st.subheader("Prediction")
    st.write(f"The uploaded image is classified as: **{predicted_class}**")
