import os
import requests
import streamlit as st
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ensure models folder exists (as a folder!)
if not os.path.exists("models"):
    os.makedirs("models")

# Download from Google Drive (file ID format)
def download_file(url, path):
    if not os.path.exists(path):
        st.info(f"Downloading {os.path.basename(path)}...")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
        st.success(f"{os.path.basename(path)} downloaded.")

# Google Drive links (use uc?id=... format)
SIM_MATRIX_URL = "https://drive.google.com/uc?id=1jHSjABmxE_1E6k7uNyewi2EM2joq3imx"
CSV_URL = "https://drive.google.com/uc?id=1GxaqWy0Lb2zO6jSlgR-GQTi2FrMEf8U0"

# Target file paths
SIM_MATRIX_PATH = "models/similarity_matrix.pkl"
CSV_PATH = "ecommerce_data_cleaned.csv"

# Download required files
download_file(SIM_MATRIX_URL, SIM_MATRIX_PATH)
download_file(CSV_URL, CSV_PATH)

# Load models from local GitHub upload
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load downloaded files
with open(SIM_MATRIX_PATH, "rb") as f:
    similarity_matrix = pickle.load(f)

df = pd.read_csv(CSV_PATH)

# --- rest of your Streamlit app code here ---
