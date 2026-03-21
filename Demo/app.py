import streamlit as st
import numpy as np
import pandas as pd
import folium
import joblib

from streamlit_folium import st_folium
from sklearn.neighbors import KNeighborsClassifier
from folium.plugins import MarkerCluster

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="ChittoYatra | Live Order Clustering Demo",
    layout="wide"
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            rgba(182, 94, 186, 0.2),
            rgba(218, 235, 253, 0.5)
        );
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<h3 style=margin-top:-60px;>ChittoYatra Live Order-to-Cluster Assignment</h3>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans.pkl"))
    dbscan = joblib.load(os.path.join(BASE_DIR, "dbscan.pkl"))
    hc = joblib.load(os.path.join(BASE_DIR, "hierarchical.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return kmeans, dbscan, hc, scaler

kmeans, dbscan, hc, scaler = load_models()

@st.cache_data
def load_sample():
    return pd.read_csv(os.path.join(BASE_DIR, "chitto_sample_5k.csv"))

df = load_sample()

if "new_orders" not in st.session_state:
    st.session_state.new_orders = []


st.sidebar.header("New Incoming Order")

model_choice = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ("K-Means", "DBSCAN", "Hierarchical")
)

delivery_lat = st.sidebar.number_input(
    "Delivery Latitude",
    float(df["delivery_gps_lat"].min()),
    float(df["delivery_gps_lat"].max()),
    float(df["delivery_gps_lat"].mean()),
    format="%.5f"
)

delivery_lng = st.sidebar.number_input(
    "Delivery Longitude",
    float(df["delivery_gps_lng"].min()),
    float(df["delivery_gps_lng"].max()),
    float(df["delivery_gps_lng"].mean()),
    format="%.5f"
)

accept_lat = st.sidebar.number_input(
    "Accept Latitude",
    float(df["accept_gps_lat"].min()),
    float(df["accept_gps_lat"].max()),
    float(df["accept_gps_lat"].mean()),
    format="%.5f"
)

accept_lng = st.sidebar.number_input(
    "Accept Longitude",
    float(df["accept_gps_lng"].min()),
    float(df["accept_gps_lng"].max()),
    float(df["accept_gps_lng"].mean()),
    format="%.5f"
)

X_new = np.array([[delivery_lat, delivery_lng, accept_lat, accept_lng]])
X_new_scaled = scaler.transform(X_new)

if model_choice == "K-Means":
    cluster_id = int(kmeans.predict(X_new_scaled)[0])
    cluster_col = "cluster_kmeans"

elif model_choice == "DBSCAN":
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(df[['delivery_gps_lat','delivery_gps_lng','accept_gps_lat','accept_gps_lng']],
            df['cluster_dbscan'])
    cluster_id = int(knn.predict(X_new)[0])
    cluster_col = "cluster_dbscan"

else:  
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(df[['delivery_gps_lat','delivery_gps_lng','accept_gps_lat','accept_gps_lng']],
            df['cluster_hc'])
    cluster_id = int(knn.predict(X_new)[0])
    cluster_col = "cluster_hc"


col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Add Order"):
        st.session_state.new_orders.append({
            "delivery_lat": delivery_lat,
            "delivery_lng": delivery_lng,
            "accept_lat": accept_lat,
            "accept_lng": accept_lng,
            "cluster": cluster_id,
            "model": model_choice
        })

with col2:
    if st.button("Reset"):
        st.session_state.new_orders = []

unique_clusters = sorted(df[cluster_col].unique())
cluster_colors = {}

for idx, cl in enumerate(unique_clusters):
    if cl == 11:
        cluster_colors[cl] = "#764E84"
    else:
        r = (idx * 70 + 180) % 256
        g = (idx * 120 + 200) % 256
        b = (idx * 180 + 220) % 256
        cluster_colors[cl] = f"#{r:02x}{g:02x}{b:02x}"

cluster_colors[-1] = "black"

st.markdown(
    f"""
    <div style="
        background-color:{cluster_colors.get(cluster_id, 'gray')};
        padding:15px;
        border-radius:10px;
        text-align:center;
        color:white;
        font-size:22px;
        font-weight:bold;
    ">
        Latest Order → Cluster {cluster_id} ({model_choice})
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

m = folium.Map(location=[df["delivery_gps_lat"].mean(), df["delivery_gps_lng"].mean()], zoom_start=12)

marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    cl = int(row[cluster_col])
    folium.CircleMarker(
        location=[row["delivery_gps_lat"], row["delivery_gps_lng"]],
        radius=3,
        color=cluster_colors.get(cl, "gray"),
        fill=True,
        fill_color=cluster_colors.get(cl, "gray"),
        fill_opacity=0.6
    ).add_to(marker_cluster)

for order in st.session_state.new_orders:
    folium.CircleMarker(
        location=[order["delivery_lat"], order["delivery_lng"]],
        radius=8,
        color="red",
        fill=True,
        fill_opacity=0.9,
        popup=f"Cluster {order['cluster']} ({order['model']})"
    ).add_to(m)

if st.session_state.new_orders:
    last = st.session_state.new_orders[-1]
    folium.Marker(
        location=[last["delivery_lat"], last["delivery_lng"]],
        popup=f"NEW ORDER → Cluster {last['cluster']}",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)

st_folium(m, width="100%", height=300)


st.subheader("Live Incoming Orders Table")

if st.session_state.new_orders:
    live_df = pd.DataFrame(st.session_state.new_orders)
    st.dataframe(live_df, use_container_width=True)
else:
    st.info("No incoming orders yet. Add a new order from the sidebar.")
