import streamlit as st
import pandas as pd
import pickle
import os
import requests

# Google Drive links (only the large files)
SIM_MATRIX_URL = "https://drive.google.com/uc?id=1jHSjABmxE_1E6k7uNyewi2EM2joq3imx"
CSV_URL = "https://drive.google.com/uc?id=1GxaqWy0Lb2zO6jSlgR-GQTi2FrMEf8U0"

# Local file paths
SIM_MATRIX_PATH = "similarity_matrix.pkl"
CSV_PATH = "ecommerce_data_cleaned.csv"

# Download helper
def download_file(url, path):
    if not os.path.exists(path):
        st.info(f"Downloading {os.path.basename(path)}...")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)
        st.success(f"{os.path.basename(path)} downloaded.")

# 📥 Download large files from Google Drive
download_file(SIM_MATRIX_URL, SIM_MATRIX_PATH)
download_file(CSV_URL, CSV_PATH)

# ✅ Load downloaded files
with open(SIM_MATRIX_PATH, "rb") as f:
    similarity_df = pickle.load(f)

# ✅ Validate that the similarity matrix is a DataFrame
if not isinstance(similarity_df, pd.DataFrame):
    st.error("Loaded similarity matrix is not a DataFrame. Please check the file.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# ✅ Load uploaded model files
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create mappings
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
