import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# Google Drive File IDs (not full links)
SIM_MATRIX_ID = "1leLLub0cDy5ENkWsIodkYR0NhoSrQpGl"
CSV_ID = "1GxaqWy0Lb2zO6jSlgR-GQTi2FrMEf8U0"

# File save paths
SIM_MATRIX_PATH = "similarity_matrix.pkl"
CSV_PATH = "ecommerce_data_cleaned.csv"

# File download helper using gdown
def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        st.info(f"Downloading {os.path.basename(output_path)}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        st.success(f"{os.path.basename(output_path)} downloaded.")

# Download the large files
download_file_from_drive(SIM_MATRIX_ID, SIM_MATRIX_PATH)
download_file_from_drive(CSV_ID, CSV_PATH)

# Load similarity matrix
try:
    with open(SIM_MATRIX_PATH, "rb") as f:
        similarity_df = pickle.load(f)
    if not isinstance(similarity_df, pd.DataFrame):
        st.error("Loaded similarity matrix is not a DataFrame.")
        st.stop()
except Exception as e:
    st.error(f"Error loading similarity matrix: {e}")
    st.stop()

# Load dataset
df = pd.read_csv(CSV_PATH)

# Load model files (should already be uploaded to repo)
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Product mappings
product_lookup = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()
product_map = df[['Description', 'StockCode']].drop_duplicates().set_index('Description')['StockCode'].to_dict()

# Streamlit UI
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("Shopper Spectrum")
st.subheader("Product Recommendation System")

product_name = st.text_input("Enter Product Name (exact)", '')

if st.button("Get Recommendations"):
    if product_name in product_map:
        product_code = product_map[product_name]
        if product_code in similarity_df.columns:
            similar_products = similarity_df[product_code].sort_values(ascending=False).drop(product_code).head(5)
            st.subheader("Top 5 Similar Products:")
            for item in similar_products.index:
                name = product_lookup.get(item, "Unknown Product")
                st.markdown(f"- **{name}** (`{item}`)")
        else:
            st.error("Product code not found in similarity matrix.")
    else:
        st.error("Product name not found.")
