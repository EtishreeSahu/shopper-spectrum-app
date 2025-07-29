import streamlit as st
import pandas as pd
import pickle

# Load cleaned dataset (for product name lookup)
df = pd.read_csv("ecommerce_data_cleaned.csv")

# Load models
with open("models/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/similarity_matrix.pkl", "rb") as f:
    similarity_df = pickle.load(f)

# Create Description → StockCode map (for input)
product_map = df[['Description', 'StockCode']].drop_duplicates().set_index('Description')['StockCode'].to_dict()

# Create StockCode → Description map (for output)
product_lookup = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()

# Streamlit page config
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("Shopper Spectrum")
st.subheader("Customer Segmentation and Product Recommendation System")

# Sidebar for module selection
option = st.sidebar.selectbox("Select Module", ["Product Recommendation", "Customer Segmentation"])

# ----------------- Product Recommendation -----------------
if option == "Product Recommendation":
    st.header("Product Recommendation")

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

# ----------------- Customer Segmentation -----------------
else:
    st.header("Customer Segmentation")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0)

    if st.button("Predict Cluster"):
        user_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(user_data)[0]

        def label_cluster(c):
            if c == 0:
                return "High-Value"
            elif c == 1:
                return "Regular"
            elif c == 2:
                return "Occasional"
            else:
                return "At-Risk"

        segment = label_cluster(cluster)
        st.success(f"Predicted Segment: {segment}")
