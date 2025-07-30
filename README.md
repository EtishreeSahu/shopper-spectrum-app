# 🛒 Shopper Spectrum

**Shopper Spectrum** is an intelligent product recommendation system that uses machine learning to suggest similar products based on user input. It analyzes product purchase patterns from an e-commerce dataset and recommends items using clustering and similarity scoring.

🎯 **Live App**: [Shopper Spectrum on Streamlit](https://shopper-spectrum-app-hqtme5uzbdcjeqnzrjz9lk.streamlit.app/)

---

## 🚀 Features

- 🔍 Search for a product name and get 5 similar recommendations
- 📊 Built using KMeans clustering and cosine similarity
- 📁 Cleaned e-commerce dataset from real-world transactions
- ☁️ Hosted on Streamlit Cloud

---

## 🧠 How It Works

1. Products are clustered using `KMeans`.
2. Product features are scaled using `StandardScaler`.
3. Cosine similarity is used to build a similarity matrix.
4. When a product is selected, top 5 similar items are shown.

---

## 📁 Project Structure

```
shopper-spectrum-app/
│
├── app.py                     # Streamlit web app
├── kmeans_model.pkl           # Trained KMeans model
├── scaler.pkl                 # Fitted StandardScaler
├── similarity_matrix.pkl      # Similarity matrix (downloaded via gdown)
├── ecommerce_data_cleaned.csv # E-commerce dataset (from Google Drive)
├── requirements.txt           # Python package dependencies
└── README.md                  # Project readme (this file)
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/etishreesahu/shopper-spectrum-app.git
cd shopper-spectrum-app
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 💻 Run the App Locally

```bash
streamlit run app.py
```

---

## 🔗 External Files (Automatically Downloaded)

These large files are hosted on Google Drive and will be downloaded when the app runs:

- `similarity_matrix.pkl`
- `ecommerce_data_cleaned.csv`

---

## 👩‍💻 Author

**Etishree Sahu**  
Aspiring AI/ML Engineer | Passionate about product intelligence & UI design  
GitHub: [@etishreesahu](https://github.com/etishreesahu)
