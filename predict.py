import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from modules.grad_cam import get_grad_cam, superimpose_grad_cam

# --- Configuration ---
MODEL_PATH = 'saved_model/plant_disease_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'

# Load the trained model and class indices
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
# Invert the dictionary to map index to class name
class_names = {v: k for k, v in class_indices.items()}

def preprocess_image(img_path):
    """Loads and preprocesses an image for model prediction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(img_path):
    """
    Predicts the class of a single image and generates Grad-CAM.
    """
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction)
    
    # Generate Grad-CAM visualization
    # For VGG16, the last convolutional layer is 'block5_conv3'
    heatmap = get_grad_cam(model, processed_img, 'block5_conv3')
    superimposed_img = superimpose_grad_cam(img_path, heatmap)
    
    return predicted_class_name, confidence, superimposed_img