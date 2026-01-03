import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import time

st.set_page_config(layout="wide", page_title="AUV Swarm Digital Twin")

st.title("🌊 Cloud-Edge AUV Swarm Digital Twin")

# Sidebar for controls
st.sidebar.header("Mission Control")
refresh_rate = st.sidebar.slider("Refresh Rate (ms)", 100, 2000, 500)
st.sidebar.success("Cloud Layer Connected")

# Placholder Data Generation (Simulating MQTT Stream)
def get_mock_data(num_agents=5):
    # This would normally come from src.edge.mqtt_bridge state buffer
    data = []
    for i in range(num_agents):
        data.append({
            "id": f"auv_{i}",
            "x": np.random.uniform(0, 100),
            "y": np.random.uniform(0, 100),
            "z": np.random.uniform(0, 50),
            "battery": np.random.uniform(20, 100),
            "status": "OK" if np.random.rand() > 0.1 else "FAULT"
        })
    return pd.DataFrame(data)

# Main Dashboard Layout
col1, col2 = st.columns([3, 1])

# 3D Visualization (Center)
with col1:
    st.subheader("Real-Time 3D Swarm View")
    
    # Get Data
    df = get_mock_data()
    
    # PyDeck Layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["x", "y", "z"],
        get_color="[200, 30, 0, 160]" if "FAULT" in df['status'].values else "[0, 255, 200, 160]",
        get_radius=2,
    )
    
    view_state = pdk.ViewState(
        latitude=0, longitude=0, zoom=1, pitch=45, bearing=0
    )
    
    # We map 100m x 100m to lat/lon for PyDeck viz (approx conversion)
    # 1 deg ~ 111km -> 100m ~ 0.001 deg
    df['lat'] = df['y'] * 0.00001
    df['lon'] = df['x'] * 0.00001
    
    layer_geo = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["lon", "lat"],
        get_fill_color=[0, 255, 0],
        get_radius=100000, # Large scale for visibility in this lat/lon hack
    )

    # Better Viz: Standard 3D Point Cloud in Local Space (Cartesian)
    # PyDeck needs Lat/Lon usually, but we can stick to simple scatter for demo
    st.dataframe(df) # Debug view

# Telemetry (Right)
with col2:
    st.subheader("Swarm Health")
    for _, row in df.iterrows():
        batt = row['battery']
        cols = st.columns([1, 2])
        cols[0].write(f"**{row['id']}**")
        cols[1].progress(int(batt))
        if row['status'] == "FAULT":
            st.error(f"{row['id']} FAULT DETECTED")

# Auto-Refresh Logic
if st.button("Start Live Monitor"):
    while True:
        time.sleep(refresh_rate/1000)
        st.experimental_rerun()
