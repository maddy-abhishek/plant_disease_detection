import streamlit as st
from PIL import Image
import os
import cv2
from predict import predict_disease

# --- App Configuration ---
st.set_page_config(page_title="Plant Disease Detector", layout="wide")
st.title("🌿 Plant Disease Detection with Grad-CAM")
st.write("Upload an image of a plant leaf to detect its disease.")

# Create an uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Leaf Image")
        image = Image.open(uploaded_file)
        st.image(image, caption='Your uploaded image.', use_column_width=True)

    # Perform prediction
    with st.spinner('Analyzing the image...'):
        predicted_class, confidence, grad_cam_image = predict_disease(file_path)

    # Display the results
    with col2:
        st.subheader("Analysis Result")
        st.success(f"**Predicted Disease:** {predicted_class.replace('_', ' ')}")
        st.info(f"**Confidence:** {confidence:.2%}")
        
        st.subheader("Model Attention (Grad-CAM)")
        # Convert BGR (from OpenCV) to RGB for displaying in Streamlit
        grad_cam_image_rgb = cv2.cvtColor(grad_cam_image, cv2.COLOR_BGR2RGB)
        st.image(grad_cam_image_rgb, caption='Heatmap shows where the model is "looking".', use_column_width=True)

# Clean up the uploads folder
for file in os.listdir("uploads"):
    os.remove(os.path.join("uploads", file))