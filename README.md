🖼️ Image Forgery Detection using Machine Learning

This project implements a lightweight image forgery detection system using traditional image processing and machine learning. It is designed as a proof-of-concept to demonstrate how handcrafted features and a simple model can detect tampered images, supported by a modern Streamlit web app for interaction.

🚩 Problem Statement

Image forgery is a growing concern in today’s digital world, as images can be manipulated easily using editing tools. Detecting such tampering is critical in fields like journalism, law enforcement, and digital forensics.
This project provides a demonstration system that can classify an uploaded image as Original or Forged, making use of feature extraction and machine learning.

🛠️ Approach

Feature Extraction with OpenCV

Grayscale histogram for intensity distribution.

Edge detection using Canny to highlight sharp transitions.

Statistical descriptors (mean, variance) for texture.

Model Training

Features were extracted from a dataset of authentic and forged images.

A Random Forest Classifier was trained to distinguish between the two classes.

Saved model (model.pkl) is used in the app.

User Interface with Streamlit

Upload an image from your local machine.

Features are extracted automatically.

The ML model predicts whether the image is Original or Forged.

🖥️ Tech Stack

Python 3.9+

OpenCV – image preprocessing

NumPy – numerical operations

Scikit-learn – machine learning model

Streamlit – interactive web UI
