# 💎 Diamond Price Prediction

A Machine Learning project that predicts diamond prices based on important characteristics such as carat, cut, color, clarity, depth, and table.

## 🎯 Objective

Build a machine learning model to predict diamond prices and compare multiple regression algorithms to identify the best-performing model.

## 📊 Dataset

The dataset contains diamond attributes including:

- Carat
- Cut
- Color
- Clarity
- Depth
- Table
- Price
- X, Y, Z dimensions

## 🔍 Machine Learning Workflow

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Analysis
5. Categorical Feature Encoding
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Model Comparison
10. Price Prediction

## 🤖 Algorithms Used

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## 🏆 Model Performance

The trained models were evaluated and compared based on their prediction performance.

**Random Forest** achieved the best performance among the evaluated models in the original project.

## 🛠️ Technologies & Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- Streamlit

## 📁 Project Structure

```text
Diamond-Price-Prediction/
│
├── Diamonds.ipynb
├── diamonds.csv
├── app.py
├── README.md
├── scaler.pkl
├── cluster_scaler.pkl
├── diamond_cluster_model.pkl
└── pca_model.pkl
