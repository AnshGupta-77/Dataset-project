# Dataset-project
This project builds an end-to-end data preprocessing and feature engineering pipeline for the Indian Flight dataset using Python. It handles missing values, incorrect data types, duplicates, outliers, and irrelevant features, and applies multiple encoding and scaling techniques to create clean, model-ready data for machine learning.

# ✈️ Indian Flight Data Preprocessing & Feature Engineering

## 📌 Project Overview
This project builds an end-to-end data preprocessing and feature engineering pipeline for the Indian Flight dataset using Python. The goal is to clean, transform, and prepare raw data into a model-ready format suitable for various machine learning algorithms.

---

## 📂 Dataset
- **Name:** Indian Flight Dataset  
- **Type:** Tabular (CSV)  
- **Contains:** Flight details such as airline, source, destination, duration, price, etc.

---

## ⚙️ Preprocessing Steps
- Handling missing values  
- Correcting incorrect data types and formats  
- Removing duplicate records  
- Detecting and handling outliers (IQR method)  
- Removing irrelevant or redundant features  

---

## 🔡 Encoding Techniques Used
- Label Encoding  
- One-Hot Encoding  
- Ordinal Encoding  
- Binary Encoding  
- Count Encoding  
- Target Encoding  

---

## 📏 Feature Scaling Methods
- Min-Max Scaling  
- Max-Abs Scaling  
- Vector Normalization  
- Z-Score (Standardization)  

---

## 🛠️ Technologies & Libraries
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- category_encoders  

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn category_encoders
python project.py
