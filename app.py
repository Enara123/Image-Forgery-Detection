import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

# Load saved model
clf = joblib.load("forgery_model.pkl")

# Feature extraction function
def extract_features(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge feature
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    
    # 2. Histogram feature (16 bins)
    hist = cv2.calcHist([gray], [0], None, [16], [0,256])
    hist = hist.flatten() / hist.sum()  # normalize
    
    # Combine all features into a single vector
    features = np.hstack([edge_count, hist])
    return features


# Streamlit UI
st.title("🖼️ Image Forgery Detection")
st.write("Upload an image and the model will predict if it is Forged or Original.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    
    # Convert to OpenCV format and resize
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (128,128))
    
    # Extract features and predict
    feat = extract_features(img_cv)
    pred = clf.predict([feat])[0]
    label = "Forged" if pred == 1 else "Original"
    
    st.success(f"Prediction: {label}")
