# 🌿 Plant Disease Detection with Transfer Learning
A deep learning project to identify plant diseases from leaf images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model is served through a user-friendly web application created with Streamlit.

This project demonstrates skills in Computer Vision, Transfer Learning, and Model Interpretability with Grad-CAM.

# ✨ Features
  - High-Accuracy Classification: Uses a pre-trained VGG16 model fine-tuned on the PlantVillage dataset for robust performance.

  - Interactive Web App: A simple and clean UI built with Streamlit to upload images and get instant predictions.

  - Model Interpretability: Implements Grad-CAM to visualize which parts of the leaf image the model focused on for its prediction, helping to build trust in the model's decisions.

  - Modular Codebase: The project is organized into logical modules for data loading, model building, training, and prediction, making it easy to understand and extend.

# 🖥️ Demo
Here's a look at the web application interface. The user uploads an image, and the app displays the prediction along with the Grad-CAM heatmap.

# 🛠️ Tech Stack
  - Backend & Model: Python, TensorFlow, Keras

  - Web Framework: Streamlit

  - Image Processing: OpenCV, Pillow

  - Dataset: PlantVillage Dataset

# 📁 Project Structure
The project is organized in a modular format for clarity and scalability.

plant-disease-detector/

│

├── app.py                   # Main web application file (Streamlit)

├── train.py                 # Script to train the model

├── predict.py               # Script for making predictions

│

├── modules/

│   ├── data_loader.py       # Handles data loading and augmentation

│   ├── model_builder.py     # Defines and builds the CNN model

│   └── grad_cam.py          # Logic for Grad-CAM visualization

│
├── saved_model/

│   └── plant_disease_model.h5 # Trained model is saved here

│

├── uploads/                   # Temporary storage for user uploads

│

├── class_indices.json       # Maps model output to class names

└── requirements.txt         # Project dependencies

- Module 1: Data Loading & Augmentation
  
This module prepares your data. It creates data generators that load images from your dataset directory, resize them, and apply augmentations in real-time to prevent overfitting.

- Module 2: Model Building (Transfer Learning)
  
This module uses a powerful pre-trained model (VGG16) and adds custom layers on top. We freeze the base model's layers to leverage its learned features and only train our new layers.

- Module 3: Training Script
  
This script ties everything together. It uses the data_loader to get the data and the model_builder to get the model, then starts the training process.

- Module 4: Prediction & Grad-CAM
  
These modules handle inference. predict.py loads the trained model and preprocesses an image for prediction. grad_cam.py generates a heatmap to visualize why the model made its prediction.

- Module 5: Web Application
  
This uses Streamlit to create a simple, interactive web app. Users can upload an image, and the app will display the prediction and the Grad-CAM visualization.

# 🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

1. Create a Virtual Environment
   
  - It's highly recommended to use a virtual environment to manage project dependencies.
    
2. Install Dependencies

  - Install all the required Python packages.
    
3. Download the Dataset

You need the PlantVillage dataset to train the model.

  - Download the dataset from this Kaggle link.

  - Extract the contents. You should have a directory structure like New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train, valid, etc

  - Copy the folder containing all the disease classes (e.g., the train folder) to a location of your choice.

  - Configure the Training Path

    Open the train.py file and update the DATASET_PATH variable to point to the directory where you saved the dataset.

    # In train.py
    
DATASET_PATH = 'path/to/your/PlantVillage/dataset' # IMPORTANT: Change this!

# How to run
1. Train the Model

   Run the training script. This will process the dataset, train the model, and save the plant_disease_model.h5 and class_indices.json files in their respective directories

   python train.py

2. Run the Web Application

   Once the model is trained and saved, start the Streamlit web application

   streamlit run app.py
