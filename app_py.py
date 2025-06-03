import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np
import pandas as pd
import h5py

# --- Configure ---
MODEL_PATH = 'skin_disease_model.h5'
SEGMENTATION_MODEL_PATH = 'SegCNN.h5'
IMAGE_HEIGHT, IMAGE_WIDTH = 75, 100

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

# --- Custom Styling ---
st.set_page_config(page_title="Melanoma Detection", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 128, 128, 0.2);
        }
        h1, h2, h3 {
            color: #007acc;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stFileUploader {
            border: 2px dashed #007acc;
            padding: 1rem;
            background-color: #e6f7ff;
            border-radius: 10px;
        }
        .stDataFrame {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# --- Fix layer names in HDF5 file ---
def sanitize_h5_model_names(file_path):
    with h5py.File(file_path, 'r+') as f:
        if 'model_weights' in f:
            for layer_name in list(f['model_weights'].keys()):
                if '/' in layer_name:
                    sanitized = layer_name.replace('/', '__')
                    f['model_weights'].move(layer_name, sanitized)
        if 'layer_names' in f.attrs:
            layer_names = [n.decode('utf-8').replace('/', '__') for n in f.attrs['layer_names']]
            f.attrs.modify('layer_names', [n.encode('utf-8') for n in layer_names])

# --- Load Models ---
@st.cache_resource
def load_keras_model(model_path):
    try:
        sanitize_h5_model_names(model_path)  # Fix layer names before loading
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img_pil):
    img_resized = img_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_mean = np.mean(img_array_expanded)
    img_std = np.std(img_array_expanded)
    return (img_array_expanded - img_mean) / img_std if img_std != 0 else img_array_expanded - img_mean

def generate_segmentation(img_pil, segmentation_model):
    img_resized = img_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = keras_image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    segmentation_mask = segmentation_model.predict(img_array_expanded)
    return segmentation_mask[0]

# --- Title Section ---
st.markdown("<h1 style='text-align: center;'>🧬 AI-Powered Skin Lesion Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload a skin lesion image to classify and segment it using AI</p>", unsafe_allow_html=True)

# --- Sidebar Upload ---
st.sidebar.header("📤 Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

# --- Load Models ---
classification_model = load_keras_model(MODEL_PATH)
segmentation_model = load_keras_model(SEGMENTATION_MODEL_PATH)

if classification_model is None or segmentation_model is None:
    st.stop()

# --- Main UI Layout ---
if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Uploaded Image")
        st.image(img_pil, caption='Original Image', use_column_width=True)

    img_processed = preprocess_image(img_pil)

    with st.spinner("🧠 Classifying..."):
        class_predictions = classification_model.predict(img_processed)
    predicted_class_index = np.argmax(class_predictions)
    predicted_class_name = label_map.get(predicted_class_index, "Unknown")
    confidence = np.max(class_predictions) * 100

    with st.spinner("🧪 Segmenting..."):
        segmentation_mask = generate_segmentation(img_pil, segmentation_model)
        segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)
        segmented_image = Image.fromarray(segmentation_mask.squeeze() * 255)

    with col2:
        st.subheader("📈 Prediction Result")
        st.success(f"**Disease:** {predicted_class_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader("🖼 Segmentation")
        st.image(segmented_image, caption="Detected Lesion Area", use_column_width=True)

        st.subheader("📊 Class Probabilities")
        class_labels = [label_map[i] for i in range(len(label_map))]
        probs_df = pd.DataFrame({
            'Class': class_labels,
            'Probability': class_predictions[0]
        }).sort_values(by='Probability', ascending=False)
        probs_df['Probability'] = probs_df['Probability'].map('{:.2%}'.format)
        st.dataframe(probs_df, use_container_width=True)

else:
    st.info("Please upload an image from the sidebar to begin.")
