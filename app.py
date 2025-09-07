import streamlit as st
from PIL import Image
import time
import requests # This would be used to call your model's API
import io
from fastai.vision.all import *
import pathlib
import platform

# Temporary fix for PosixPath issue on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# --- Page Configuration ---
st.set_page_config(
    page_title="AgriScan",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Custom CSS for Styling ---
# This CSS helps in mimicking the look and feel of the provided UI design.
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #F0F2F6;
    }

    /* Header styling */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 2rem;
    }
    .header .title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.75rem;
        font-weight: bold;
        color: #212529; /* Changed title color to be visible */
    }

    /* Card styling */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1.5rem;
    }

    /* Custom button styling */
    div.stButton > button {
        background-color: #16A34A;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #15803D;
    }
    div.stButton > button:disabled {
        background-color: #A3A3A3;
        color: #F5F5F5;
    }

</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
# Use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_model():
    """Load the fastai learner model."""
    try:
        # IMPORTANT: Ensure this path matches the name of your exported model file.
        model_path = "resnet50_bovine.pkl"
        # --- THE FIX IS HERE: added cpu=True ---
        # This tells fastai to load the model onto the CPU.
        model = load_learner(model_path, cpu=True)
        return model
    except FileNotFoundError:
        # This error is critical and should be shown clearly.
        st.error(f"Model file '{model_path}' not found. Please make sure it's in the correct folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None


model = load_model()


# --- Header ---
st.markdown("""
<div class="header">
    <div class="title">
        <span>🌿</span>
        <span>AgriScan</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Image Upload Section ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Cattle/Buffalo Photo",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        # Show this message only if the model has loaded successfully
        if model:
            st.info("Please upload an image file to predict the breed.")
    st.markdown('</div>', unsafe_allow_html=True)


# --- Aadhar Input and Submission ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    aadhar_number = st.text_input("Aadhar Number", placeholder="12-digit number", max_chars=12)

    # Disable button if model isn't loaded or inputs are missing
    is_disabled = (uploaded_file is None or not aadhar_number or model is None)
    predict_button = st.button("Predict Breed", disabled=is_disabled)
    st.markdown('</div>', unsafe_allow_html=True)


# --- Prediction Logic ---
if predict_button:
    if not aadhar_number.isdigit() or len(aadhar_number) != 12:
        st.error("Please enter a valid 12-digit Aadhar number.")
    else:
        with st.spinner('Analyzing image... Please wait.'):
            # Convert uploaded file to bytes for prediction
            img_bytes = uploaded_file.getvalue()

            # --- ACTUAL PREDICTION LOGIC with CONFIDENCE ---
            try:
                # Use the loaded fastai model to predict
                pred_class, pred_idx, outputs = model.predict(img_bytes)
                # Get the confidence score for the predicted class
                confidence = outputs[pred_idx].item() * 100

                # Display the result
                st.success(f"**Prediction Result:** The predicted breed is **{pred_class}**.")
                st.info(f"**Confidence:** {confidence:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

