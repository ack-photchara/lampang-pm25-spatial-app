import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Lampang PM2.5 Spatial Monitoring",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ ระบบแผนที่ติดตาม PM2.5 เชิงพื้นที่ความละเอียดสูง 225 จุด (อ.เมืองลำปาง)")
st.caption("ระบบจำลองและคาดการณ์ตามโครงสร้างภูมิประเทศแอ่งกระทะ (Two-Stage Spatial Machine Learning Framework)")

# 1. ดึงสภาพอากาศสดจาก Open-Meteo API (Cache 10 นาที)
@st.cache_data(ttl=600)
def fetch_live_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=18.2888&longitude=99.5056&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    try:
        res = requests.get(url, timeout=5).json()
        curr = res.get("current", {})
        return {
            "temp": float(curr.get("temperature_2m", 28.5)),
            "rh": float(curr.get("relative_humidity_2m", 65.0)),
            "wind": float(curr.get("wind_speed_10m", 5.2)),
            "status": "Online (Live API)"
        }
    except Exception:
        return {"temp": 28.5, "rh": 65.0, "wind": 5.2, "status": "Fallback Baseline"}

# 2. โหลดตารางกริดความสูงจริง
@st.cache_data
def load_grid_data():
    try:
        df = pd.read_csv("grid_lampang.csv")
        return df
    except Exception:
        # Fallback กรณีไม่มีไฟล์
        lats = np.linspace(18.20, 18.35, 15)
        lons = np.linspace(99.40, 99.55, 15)
        records = []
        for i, (lat, lon) in enumerate(zip(np.repeat(lats, 15), np.tile(lons, 15))):
            records.append({"grid_id": f"GRID_{i+1:03d}", "lat": lat, "lon": lon, "elevation": 240.0})
        return pd.DataFrame(records)

live_weather = fetch_live_weather()
df_grid = load_grid_data().copy()

# 3. Sidebar ควบคุมสถานการณ์
st.sidebar.header("🎛️ แผงควบคุมและสถานการณ์จำลอง")
app_mode = st.sidebar.radio("เลือกโหมด:", ["📡 ข้อมูลสภาพอากาศสด (Live)", "🧪 สถานการณ์จำลอง (What-If)"])

if app_mode == "📡 ข้อมูลสภาพอากาศสด (Live)":
    st.sidebar.success(f"สถานะ: {live_weather['status']}")
    temp_val = live_weather["temp"]
    rh_val = live_weather["rh"]
    wind_val = live_weather["wind"]
    pm_raw_val = 55.0
else:
    preset = st.sidebar.selectbox(
        "เลือก Preset ด่วน:",
        ["กำหนดค่าเอง (Custom)", "🔴 วิกฤตหมอกควันปิดเมือง", "🟢 ลมพัดระบายหลังฝนตก"]
    )
    if preset == "🔴 วิกฤตหมอกควันปิดเมือง":
        d_temp, d_rh, d_wind, d_pm = 32.0, 85.0, 1.2, 75.0
    elif preset == "🟢 ลมพัดระบายหลังฝนตก":
        d_temp, d_rh, d_wind, d_pm = 24.0, 45.0, 18.0, 25.0
    else:
        d_temp, d_rh, d_wind, d_pm = live_weather["temp"], live_weather["rh"], live_weather["wind"], 55.0

    temp_val = st.sidebar.slider("อุณหภูมิ (°C)", 15.0, 45.0, float(d_temp))
    rh_val = st.sidebar.slider("ความชื้นสัมพัทธ์ RH (%)", 10.0, 100.0, float(d_rh))
    wind_val = st.sidebar.slider("ความเร็วลม (km/h)", 0.0, 30.0, float(d_wind))
    pm_raw_val = st.sidebar.number_input("ค่าฝุ่นดิบ DustBoy เฉลี่ย (µg/m³)", value=float(d_pm))

# 4. Two-Stage Machine Learning Formula
# Stage 1: สอบเทียบค่าเซนเซอร์
pm25_calibrated = 44.382 + (0.839 * pm_raw_val) - (0.357 * rh_val) + (0.151 * temp_val) - (0.099 * wind_val)

# Stage 2: Spatial Downscaling (Elevation-driven)
min_elev = df_grid["elevation"].min()
df_grid["pm25_pred"] = pm25_calibrated - ((df_grid["elevation"] - min_elev) * 0.08)
df_grid["pm25_pred"] = df_grid["pm25_pred"].clip(lower=15.0)

# แสดง KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("อุณหภูมิที่ใช้", f"{temp_val:.1f} °C")
c2.metric("ความชื้นสัมพัทธ์ (RH)", f"{rh_val:.1f} %")
c3.metric("ค่าฝุ่น Calibrated", f"{pm25_calibrated:.2f} µg/m³")
c4.metric("ช่วงค่าฝุ่นบนกริด", f"{df_grid['pm25_pred'].min():.1f} - {df_grid['pm25_pred'].max():.1f} µg/m³")

# 5. ฟังก์ชันจัดสี
def get_color(pm):
    if pm >= 59.43: return "#FF0000"
    elif pm >= 57.90: return "#FF8C00"
    elif pm >= 55.81: return "#FFD700"
    else: return "#2ECC71"

# 6. สร้างแผนที่ Folium
m = folium.Map(location=[18.275, 99.475], zoom_start=11, tiles="CartoDB positron")

for _, row in df_grid.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=7,
        color=get_color(row["pm25_pred"]),
        fill=True,
        fill_color=get_color(row["pm25_pred"]),
        fill_opacity=0.85,
        weight=1,
        popup=f"<b>{row['grid_id']}</b><br>PM2.5: {row['pm25_pred']:.2f} µg/m³<br>Elevation: {row['elevation']:.1f} m"
    ).add_to(m)

st_folium(m, width="100%", height=550)