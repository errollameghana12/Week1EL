import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors


st.set_page_config(page_title="EV Recommendation App", layout="wide")

st.title("⚡ Electric Vehicle Recommendation & Range Prediction App")


df = pd.read_csv("Cleaned_Electric_Vehicle_Data.csv")

model = joblib.load("Best_EV_Range_Model.pkl")

st.success("Dataset & Model Loaded Successfully!")



rec_df = df[['Make', 'Model Year', 'Electric Vehicle Type', 'Electric Range', 'Base MSRP']].copy()
rec_df.dropna(inplace=True)

enc_make = LabelEncoder()
enc_type = LabelEncoder()

rec_df['Make'] = enc_make.fit_transform(rec_df['Make'])
rec_df['Electric Vehicle Type'] = enc_type.fit_transform(rec_df['Electric Vehicle Type'])

scaler = MinMaxScaler()
scaled_numeric = scaler.fit_transform(rec_df[['Model Year', 'Electric Range', 'Base MSRP']])

final_features = pd.DataFrame(scaled_numeric,
                              columns=['Model Year', 'Electric Range', 'Base MSRP'])
final_features['Make'] = rec_df['Make']
final_features['Electric Vehicle Type'] = rec_df['Electric Vehicle Type']

nn_model = NearestNeighbors(metric='cosine', n_neighbors=6)
nn_model.fit(final_features)


def recommend_similar_ev(index, n=5):
    distances, indices = nn_model.kneighbors([final_features.iloc[index]], n_neighbors=n+1)
    top_indices = indices[0][1:]  # Exclude itself
    return df.iloc[top_indices][[
        'Make', 'Model Year', 'Electric Vehicle Type',
        'Electric Range', 'Base MSRP'
    ]]


def recommend_by_preferences(min_range, max_budget, ev_type):
    filtered = df[
        (df['Electric Range'] >= min_range) &
        (df['Base MSRP'] <= max_budget) &
        (df['Electric Vehicle Type'].str.contains(ev_type, case=False))
    ]
    if filtered.empty:
        return "❌ No EVs match your preferences."
    
    return filtered.head(10)[[
        'Make', 'Model Year', 'Electric Vehicle Type',
        'Electric Range', 'Base MSRP'
    ]]


tab1, tab2, tab3 = st.tabs([
    "🔮 Predict EV Range",
    "🚗 Similar EV Recommendation",
    "🎯 Recommend by Preferences"
])


with tab1:
    st.subheader("🔮 Predict EV Range using ML Model")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Model Year", min_value=2000, max_value=2025, value=2020)
        price = st.number_input("Base MSRP ($)", min_value=0, max_value=150000, value=40000)

    with col2:
        make = st.selectbox("Select Manufacturer", df['Make'].unique())
        ev_type = st.selectbox("Select EV Type", df['Electric Vehicle Type'].unique())

    if st.button("Predict Range"):
        temp = pd.DataFrame([[year, make, ev_type, price]],
                            columns=['Model Year', 'Make', 'Electric Vehicle Type', 'Base MSRP'])

        # Encode
        temp['Make'] = enc_make.transform([make])[0]
        temp['Electric Vehicle Type'] = enc_type.transform([ev_type])[0]

        prediction = model.predict(temp)[0]

        st.success(f"🔋 Estimated Range: **{round(prediction, 2)} miles**")


with tab2:
    st.subheader("🚗 Find Similar Electric Vehicles")

    index = st.number_input("Enter EV Index (0 to {}):".format(len(df)-1),
                            min_value=0, max_value=len(df)-1, value=10)

    if st.button("Show Similar EVs"):
        st.write(recommend_similar_ev(index))


with tab3:
    st.subheader("🎯 Recommend EVs Based on Your Preferences")

    min_range = st.slider("Minimum Range (miles)", 0, 400, 200)
    max_budget = st.slider("Maximum Budget ($)", 10000, 150000, 50000)
    ev_type_pref = st.selectbox("EV Type Preference", df['Electric Vehicle Type'].unique())

    if st.button("Recommend EVs"):
        st.write(recommend_by_preferences(min_range, max_budget, ev_type_pref))
