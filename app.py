import streamlit as st
import numpy as np
import pickle

# ===============================
# Constants
# ===============================
AVG_PRICE_PER_CARAT = 5500   # USD (dataset average)
USD_TO_INR = 83

# ===============================
# Load models & scalers
# ===============================
with open('final_model.pkl', 'rb') as f:
    price_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    price_scaler = pickle.load(f)

with open('kmeans_market_segmentation.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('cluster_scaler.pkl', 'rb') as f:
    cluster_scaler = pickle.load(f)

# ===============================
# Cluster name mapping
# ===============================
cluster_names = {
    0: 'Premium Heavy Diamonds',
    1: 'Affordable Small Diamonds',
    2: 'Mid-range Balanced Diamonds'
}

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Diamond Price & Market Segmentation")
st.title("💎 Diamond Price Prediction & Market Segmentation")
st.markdown("---")

# ===============================
# Sidebar
# ===============================
module = st.sidebar.selectbox(
    "Select Module",
    ["Price Prediction", "Market Segment Prediction"]
)

# ===============================
# Input Form
# ===============================
st.subheader("Enter Diamond Details")

carat = st.number_input(
    "Carat (0.1 – 5.0)",
    min_value=0.1,
    max_value=5.0,
    value=0.9,
    step=0.01
)

depth = st.number_input(
    "Depth % (50 – 70)",
    min_value=50.0,
    max_value=70.0,
    value=61.0,
    step=0.1
)

table = st.number_input(
    "Table % (50 – 70)",
    min_value=50.0,
    max_value=70.0,
    value=56.0,
    step=0.1
)

dimension_ratio = st.number_input(
    "Dimension Ratio (0.8 – 1.3)",
    min_value=0.8,
    max_value=1.3,
    value=1.05,
    step=0.01
)

cut = st.selectbox(
    "Cut (Quality Order)",
    ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
)

color = st.selectbox(
    "Color (J = Lowest, D = Best)",
    ['J', 'I', 'H', 'G', 'F', 'E', 'D']
)

clarity = st.selectbox(
    "Clarity (I1 = Lowest, IF = Best)",
    ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
)

# ===============================
# Ordinal Encoding
# ===============================
cut_map = {'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4}
color_map = {'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5, 'D':6}
clarity_map = {'I1':0, 'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7}

cut_enc = cut_map[cut]
color_enc = color_map[color]
clarity_enc = clarity_map[clarity]

# ===============================
# PRICE PREDICTION (USD → INR)
# ===============================
if module == "Price Prediction":
    if st.button("Predict Price 💰"):
        X_price = np.array([[
            carat,
            cut_enc,
            color_enc,
            clarity_enc,
            depth,
            table,
            AVG_PRICE_PER_CARAT,   # hidden feature
            dimension_ratio
        ]])

        X_price_scaled = price_scaler.transform(X_price)
        price_usd = price_model.predict(X_price_scaled)[0]
        price_inr = price_usd * USD_TO_INR

        st.success(f"💎 Predicted Diamond Price: ₹ {price_inr:,.2f}")

# ===============================
# MARKET SEGMENT PREDICTION
# ===============================
if module == "Market Segment Prediction":
    if st.button("Predict Market Segment 📊"):
        X_cluster = np.array([[
            carat,
            cut_enc,
            color_enc,
            clarity_enc,
            AVG_PRICE_PER_CARAT,
            dimension_ratio
        ]])

        X_cluster_scaled = cluster_scaler.transform(X_cluster)
        cluster_id = kmeans_model.predict(X_cluster_scaled)[0]

        st.info(f"🔢 Cluster Number: {cluster_id}")
        st.success(f"🏷 Market Segment: {cluster_names[cluster_id]}")
