import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODELS ----------------
price_model = pickle.load(open("best_price_model.pkl", "rb"))
cluster_model = pickle.load(open("diamond_cluster_model.pkl", "rb"))
cluster_scaler = pickle.load(open("cluster_scaler.pkl", "rb"))

# ---------------- CLUSTER NAMES ----------------
cluster_names = {
    0: "Affordable Small Diamonds",
    1: "Mid-range Balanced Diamonds",
    2: "Premium Heavy Diamonds"
}

# ---------------- APP UI ----------------
st.set_page_config(page_title="Diamond Analytics", layout="centered")
st.title("💎 Diamond Price Prediction & Market Segmentation")

st.markdown("Predict **diamond price** and identify its **market segment**")

# ---------------- USER INPUT ----------------
st.header("🔢 Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, step=0.01)
depth = st.number_input("Depth", step=0.1)
table = st.number_input("Table", step=0.1)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

x = st.number_input("Length (x)", step=0.1)
y = st.number_input("Width (y)", step=0.1)
z = st.number_input("Depth (z)", step=0.1)

# ---------------- ENCODING ----------------
cut_map = {"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}
color_map = {"D":0,"E":1,"F":2,"G":3,"H":4,"I":5,"J":6}
clarity_map = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}

cut = cut_map[cut]
color = color_map[color]
clarity = clarity_map[clarity]

# ---------------- FEATURE ENGINEERING ----------------
volume = x * y * z
dimension_ratio = (x + y) / (2 * z) if z != 0 else 0
log_carat = np.log1p(carat)
log_volume = np.log1p(volume)

# Dummy placeholders (used during training)
price_per_carat = 1
price_density = 1

# ---------------- INPUT DATAFRAME ----------------
input_df = pd.DataFrame([[
    carat, cut, color, clarity, depth, table,
    log_carat, volume, price_per_carat,
    dimension_ratio, log_volume, price_density
]], columns=[
    'carat','cut','color','clarity','depth','table',
    'log_carat','volume','price_per_carat',
    'dimension_ratio','log_volume','price_density'
])

# ---------------- PRICE PREDICTION ----------------
if st.button("💰 Predict Price"):
    predicted_price = price_model.predict(input_df)[0]
    st.success(f"Estimated Diamond Price: ₹ {predicted_price:,.2f}")

# ---------------- CLUSTER PREDICTION ----------------
if st.button("📊 Predict Market Segment"):
    scaled_data = cluster_scaler.transform(input_df)
    cluster = cluster_model.predict(scaled_data)[0]
    st.info(f"Market Segment: **{cluster_names[cluster]}**")
