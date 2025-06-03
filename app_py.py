import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np
import pandas as pd
import os

# --- Configuration ---
MODEL_PATH = 'skin_disease_model.h5' # Make sure this file is in the same directory as app.py
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 100
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3

# --- Label Mapping (Crucial: Ensure this matches your training!) ---
# Based on the output of notebook cell 5 - Verify this order!
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
@st.cache_resource # Cache the model loading
def load_keras_model(model_path):
    """Loads the Keras model from the specified path."""
    try:
        # Load the model without compiling it again for inference
        model = load_model(model_path, compile=False)
        # Optional: If you need to compile for specific metrics during inference (usually not needed)
        # model.compile(optimizer='adam', # Or the optimizer used during training
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Make sure 'skin_disease_model.h5' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image(img_pil):
    """Preprocesses the uploaded PIL image for the model."""
    # Resize image - PIL resize uses (Width, Height)
    img_resized = img_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert to numpy array
    img_array = keras_image.img_to_array(img_resized)

    # Add batch dimension
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Normalize using the method from the notebook *prediction* cell (cell 29)
    # This normalizes each image individually based on its own mean/std.
    # Note: This is different from the normalization in notebook cell 14.
    # Using the training set's mean/std (if saved) would be more standard,
    # but we replicate the notebook's prediction cell logic here.
    img_mean = np.mean(img_array_expanded)
    img_std = np.std(img_array_expanded)

    if img_std == 0:
        # Handle potential division by zero (e.g., solid color image)
        img_normalized = img_array_expanded - img_mean
    else:
        img_normalized = (img_array_expanded - img_mean) / img_std

    return img_normalized

# --- Streamlit App ---
st.set_page_config(page_title="Skin Disease Classifier", layout="wide")
st.title("🔬 Skin Disease Classifier")
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

# Load the model
model = load_keras_model(MODEL_PATH)

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if model is None:
    st.stop() # Stop execution if the model failed to load

if uploaded_file is not None:
    # --- Image Display and Preprocessing ---
    try:
        img_pil = Image.open(uploaded_file)

        with col1:
            st.subheader("Uploaded Image")
            st.image(img_pil, caption='Your Uploaded Image.', use_column_width=True)

        # Preprocess the image
        img_processed = preprocess_image(img_pil)

        # --- Prediction ---
        with st.spinner('Analyzing the image...'):
            predictions = model.predict(img_processed)

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = label_map.get(predicted_class_index, "Unknown Class")
        confidence = np.max(predictions) * 100

        # --- Display Results ---
        with col2:
            st.subheader("Prediction Results")
            st.success(f"**Predicted Disease:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Display probabilities for all classes
            st.write("Prediction Probabilities:")
            # Ensure labels are retrieved correctly even if prediction array is shorter/longer
            class_labels = [label_map.get(i, f"Class {i}") for i in range(NUM_CLASSES)]
            probs_df = pd.DataFrame({
                'Class': class_labels,
                'Probability': predictions[0][:NUM_CLASSES] # Ensure we only take relevant probs
            })
            probs_df['Probability'] = probs_df['Probability'].map('{:.2%}'.format) # Format as percentage
            st.dataframe(probs_df.sort_values(by='Probability', ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during image processing or prediction: {e}")
        st.error("Please try uploading a different image file.")

else:
    with col1:
        st.info("Please upload an image file using the sidebar to get a prediction.")