import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import joblib
from streamlit_folium import st_folium

st.set_page_config(page_title="Lampang PM2.5 Spatial Monitoring", page_icon="🌫️", layout="wide")
st.title("🌫️ ระบบแผนที่ติดตาม PM2.5 เชิงพื้นที่ 225 จุด (อ.เมืองลำปาง)")
st.caption("Random Forest Regression Model (10-Fold CV Aligned)")

# 1. โหลดโมเดล .pkl
@st.cache_resource
def load_rf_model():
    return joblib.load("rf_stage1_model.pkl")

rf_model = load_rf_model()

# 2. ดึงค่าฝุ่นจริงจาก Air4Thai (สถานี 35t)
@st.cache_data(ttl=600)
def fetch_live_air4thai(station_code="35t"):
    url = "http://air4thai.pcd.go.th/forappV2/getAQI_JSON.php"
    try:
        res = requests.get(url, timeout=5).json()
        for station in res.get("stations", []):
            if station.get("stationID") == station_code:
                pm_val = float(station.get("LastUpdate", {}).get("PM25", {}).get("value", 12.0))
                time_str = station.get("LastUpdate", {}).get("date") + " " + station.get("LastUpdate", {}).get("time")
                return {"pm25": pm_val, "station_name": station.get("nameTH"), "datetime": time_str}
        return {"pm25": 12.0, "station_name": "สถานีพระบาท ลำปาง", "datetime": "ล่าสุด"}
    except Exception:
        return {"pm25": 12.0, "station_name": "สถานีพระบาท ลำปาง", "datetime": "ล่าสุด"}

# 3. ดึงสภาพอากาศสด Open-Meteo
@st.cache_data(ttl=600)
def fetch_live_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=18.2888&longitude=99.5056&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    try:
        res = requests.get(url, timeout=5).json()
        curr = res.get("current", {})
        return {
            "temp": float(curr.get("temperature_2m", 28.5)),
            "rh": float(curr.get("relative_humidity_2m", 65.0)),
            "wind": float(curr.get("wind_speed_10m", 5.2))
        }
    except Exception:
        return {"temp": 28.5, "rh": 65.0, "wind": 5.2}

# 4. โหลดตารางกริด 225 จุด
@st.cache_data
def load_grid_data():
    return pd.read_csv("grid_lampang.csv")

live_air = fetch_live_air4thai("35t")
live_weather = fetch_live_weather()
df_grid = load_grid_data().copy()

# 5. Sidebar โหมดการทำงาน
st.sidebar.header("🎛️ แผงควบคุม")
app_mode = st.sidebar.radio("เลือกโหมด:", ["📡 ข้อมูลสด (Real-Time Live)", "🧪 จำลองสถานการณ์ (What-If Simulator)"])

if app_mode == "📡 ข้อมูลสด (Real-Time Live)":
    st.sidebar.success(f"เชื่อมต่อ: {live_air['station_name']}")
    pm_input = float(live_air["pm25"])
    temp_val = float(live_weather["temp"])
    rh_val = float(live_weather["rh"])
    wind_val = float(live_weather["wind"])
else:
    temp_val = st.sidebar.slider("อุณหภูมิ (°C)", 15.0, 45.0, float(live_weather["temp"]))
    rh_val = st.sidebar.slider("ความชื้นสัมพัทธ์ RH (%)", 10.0, 100.0, float(live_weather["rh"]))
    wind_val = st.sidebar.slider("ความเร็วลม (km/h)", 0.0, 30.0, float(live_weather["wind"]))
    pm_input = st.sidebar.number_input("ค่าฝุ่นฐาน (µg/m³)", value=float(live_air["pm25"]))

# 6. ส่งเข้าโมเดล Random Forest ทำนายทั้ง 225 จุด
features_matrix = pd.DataFrame({
    'pm25_raw': pm_input,
    'rh': rh_val,
    'temp': temp_val,
    'wind_speed': wind_val,
    'elevation': df_grid['elevation']
})

df_grid['pm25_pred'] = rf_model.predict(features_matrix)

# 7. แสดง KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("ค่าฝุ่นฐานสถานี", f"{pm_input:.1f} µg/m³")
c2.metric("อุณหภูมิ", f"{temp_val:.1f} °C")
c3.metric("ความชื้น (RH)", f"{rh_val:.1f} %")
c4.metric("ช่วงค่าฝุ่นบนกริด", f"{df_grid['pm25_pred'].min():.1f} - {df_grid['pm25_pred'].max():.1f} µg/m³")

# 8. พล็อตแผนที่ OpenStreetMap (ไม่มีลายน้ำ)
def get_aqi_color(pm):
    if pm >= 50.0: return "#FF0000"
    elif pm >= 37.5: return "#FF8C00"
    elif pm >= 25.0: return "#FFD700"
    elif pm >= 15.0: return "#2ECC71"
    else: return "#00BFFF"

m = folium.Map(location=[18.275, 99.475], zoom_start=11, tiles="OpenStreetMap")
for _, row in df_grid.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=7,
        color=get_aqi_color(row["pm25_pred"]),
        fill=True,
        fill_color=get_aqi_color(row["pm25_pred"]),
        fill_opacity=0.85,
        weight=1,
        popup=f"<b>{row['grid_id']}</b><br>PM2.5: {row['pm25_pred']:.2f} µg/m³<br>Elevation: {row['elevation']:.1f} m"
    ).add_to(m)

st_folium(m, width="100%", height=550)