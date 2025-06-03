
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np
import pandas as pd
import os

# --- Configuration ---
MODEL_PATH = 'D:\fyp\ui\skin_disease_model.h5'  # Classification model
SEGMENTATION_MODEL_PATH = 'D:\fyp\ui\SegCNN.h5'  # Segmentation model
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 100
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3

# --- Label Mapping (Crucial: Ensure this matches your training!) ---
label_map = {
    0: 'pigmented benign keratosis',
    1: 'melanoma',
    2: 'vascular lesion',
    3: 'actinic keratosis',
    4: 'squamous cell carcinoma',
    5: 'basal cell carcinoma',
    6: 'seborrheic keratosis',
    7: 'dermatofibroma',
    8: 'nevus'
}
NUM_CLASSES = len(label_map)

# --- Model Loading (Cached for efficiency) ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model from the specified path."""
    try:
        model = load_model(model_path, compile=False)
        st.success(f"Model loaded successfully from {model_path}!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image(img_pil):
    """Preprocesses the uploaded PIL image for the model."""
    img_resized = img_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_mean = np.mean(img_array_expanded)
    img_std = np.std(img_array_expanded)
    img_normalized = (img_array_expanded - img_mean) / img_std if img_std != 0 else img_array_expanded - img_mean
    return img_normalized

# --- Segmentation Output ---
def generate_segmentation(img_pil, segmentation_model):
    """Generates segmentation mask using the trained segmentation model."""
    img_resized = img_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    segmentation_mask = segmentation_model.predict(img_array_expanded)
    return segmentation_mask[0]

# --- Streamlit App ---
st.set_page_config(page_title="Skin Disease Classifier", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 Skin Disease Classifier")

# Customizing the theme colors
st.markdown("""
    <style>
        .css-18e3th9 {
            background-color: #4CAF50;
            color: white;
        }
        .css-1v0mbdj {
            background-color: #f4f6f9;
        }
        .css-1v0mbdj h2 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
Upload an image of a skin lesion. The application will use a pre-trained DenseNet201 model
to predict the type of skin disease from the following 9 classes based on the ISIC dataset:
*   Actinic Keratosis
*   Basal Cell Carcinoma
*   Dermatofibroma
*   Melanoma
*   Nevus
*   Pigmented Benign Keratosis
*   Seborrheic Keratosis
*   Squamous Cell Carcinoma
*   Vascular Lesion
""")

# Load the models
classification_model = load_keras_model(MODEL_PATH)
segmentation_model = load_keras_model(SEGMENTATION_MODEL_PATH)

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if classification_model is None or segmentation_model is None:
    st.stop()  # Stop execution if models failed to load

if uploaded_file is not None:
    # Image display and preprocessing
    try:
        img_pil = Image.open(uploaded_file)

        with col1:
            st.subheader("Uploaded Image")
            st.image(img_pil, caption='Your Uploaded Image.', use_column_width=True)

        # Preprocess the image
        img_processed = preprocess_image(img_pil)

        # --- Classification Prediction ---
        with st.spinner('Classifying the image...'):
            class_predictions = classification_model.predict(img_processed)
        predicted_class_index = np.argmax(class_predictions, axis=1)[0]
        predicted_class_name = label_map.get(predicted_class_index, "Unknown Class")
        confidence = np.max(class_predictions) * 100

        # --- Segmentation Prediction ---
        with st.spinner('Segmenting the image...'):
            segmentation_mask = generate_segmentation(img_pil, segmentation_model)
            segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)  # Binarize the mask
            segmented_image = Image.fromarray(segmentation_mask * 255)

        # --- Display Results ---
        with col2:
            st.subheader("Prediction Results")
            st.success(f"Predicted Disease: {predicted_class_name}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.subheader("Segmentation Output")
            st.image(segmented_image, caption="Skin Lesion Segmentation", use_column_width=True)

            # Display prediction probabilities for all classes
            st.write("Prediction Probabilities:")
            class_labels = [label_map.get(i, f"Class {i}") for i in range(NUM_CLASSES)]
            probs_df = pd.DataFrame({
                'Class': class_labels,
                'Probability': class_predictions[0][:NUM_CLASSES]  # Ensure we only take relevant probs
            })
            probs_df['Probability'] = probs_df['Probability'].map('{:.2%}'.format)  # Format as percentage
            st.dataframe(probs_df.sort_values(by='Probability', ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")
        st.error("Please try uploading a different image file.")

else:
    with col1:
        st.info("Please upload an image file using the sidebar to get a prediction.")