import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import joblib
from streamlit_folium import st_folium

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Lampang PM2.5 Spatial Monitoring",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ ระบบแผนที่ติดตาม PM2.5 เชิงพื้นที่ 225 จุด (อ.เมืองลำปาง)")
st.caption("ระบบจำลองและคาดการณ์ตามโครงสร้างภูมิประเทศแอ่งกระทะ (Two-Stage Machine Learning Framework)")

# -------------------------------------------------------------
# 2. Model & Data Loading Functions (With Caching)
# -------------------------------------------------------------
@st.cache_resource
def load_rf_model():
    """โหลดโมเดล Random Forest Regressor ที่ผ่านการเทรน 10-Fold CV"""
    try:
        return joblib.load("rf_stage1_model.pkl")
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_live_air4thai(station_code="35t"):
    """ดึงค่าฝุ่นจริงรายชั่วโมงจากกรมควบคุมมลพิษ (สถานี 35t อุตุนิยมวิทยาลำปาง พระบาท)"""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "http://air4thai.pcd.go.th/forappV2/getAQI_JSON.php"
    try:
        res = requests.get(url, headers=headers, timeout=5).json()
        for station in res.get("stations", []):
            if station.get("stationID") == station_code:
                last_up = station.get("LastUpdate", {})
                pm_val = float(last_up.get("PM25", {}).get("value", 10.3))
                date_str = last_up.get("date", "")
                time_str = last_up.get("time", "")
                return {
                    "pm25": pm_val,
                    "station_name": station.get("nameTH", "สถานีพระบาท อ.เมือง ลำปาง"),
                    "datetime": f"{date_str} {time_str}",
                    "status": "Air4Thai Online (Live)"
                }
        return {"pm25": 10.3, "station_name": "สถานีพระบาท อ.เมือง ลำปาง", "datetime": "ล่าสุด", "status": "Fallback Baseline"}
    except Exception:
        return {"pm25": 10.3, "station_name": "สถานีพระบาท อ.เมือง ลำปาง", "datetime": "ล่าสุด", "status": "Offline"}

@st.cache_data(ttl=300)
def fetch_live_weather():
    """ดึงสภาพอากาศสด ณ อ.เมืองลำปาง จาก Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast?latitude=18.2888&longitude=99.5056&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    try:
        res = requests.get(url, timeout=5).json()
        curr = res.get("current", {})
        return {
            "temp": float(curr.get("temperature_2m", 25.8)),
            "rh": float(curr.get("relative_humidity_2m", 83.0)),
            "wind": float(curr.get("wind_speed_10m", 5.2)),
            "status": "Weather Online"
        }
    except Exception:
        return {"temp": 25.8, "rh": 83.0, "wind": 5.2, "status": "Weather Offline"}

@st.cache_data
def load_grid_data():
    """โหลดตารางกริดพิกัด 225 จุดพร้อมความสูงจริง (Digital Elevation Model)"""
    try:
        return pd.read_csv("grid_lampang.csv")
    except Exception:
        # Fallback กรณีไม่มีไฟล์
        lats = np.linspace(18.20, 18.35, 15)
        lons = np.linspace(99.40, 99.55, 15)
        records = []
        for i, (lat, lon) in enumerate(zip(np.repeat(lats, 15), np.tile(lons, 15))):
            records.append({"grid_id": f"GRID_{i+1:03d}", "lat": lat, "lon": lon, "elevation": 240.0})
        return pd.DataFrame(records)

rf_model = load_rf_model()
live_air = fetch_live_air4thai("35t")
live_weather = fetch_live_weather()
df_grid = load_grid_data().copy()

# -------------------------------------------------------------
# 3. Sidebar UI & Application Controls
# -------------------------------------------------------------
st.sidebar.header("🎛️ แผงควบคุมและสถานการณ์")

app_mode = st.sidebar.radio(
    "เลือกโหมดการทำงาน:",
    ["📡 แสดงผลข้อมูลสด (Real-Time Live)", "🧪 จำลองสถานการณ์ (What-If Simulator)"]
)

if app_mode == "📡 แสดงผลข้อมูลสด (Real-Time Live)":
    st.sidebar.success(f"เชื่อมต่อ: {live_air['status']}")
    st.sidebar.caption(f"📍 {live_air['station_name']}\n\n🕒 {live_air['datetime']}")
    
    pm_base_ref = float(live_air["pm25"])
    temp_val = float(live_weather["temp"])
    rh_val = float(live_weather["rh"])
    wind_val = float(live_weather["wind"])

else:
    st.sidebar.subheader("🧪 What-If Controls")
    preset = st.sidebar.selectbox(
        "เลือก Preset สถานการณ์:",
        ["กำหนดค่าเอง (Custom)", "🔴 วิกฤตหมอกควันปิดเมือง (Stagnation)", "🟢 ลมพัดระบายหลังฝน (Ventilation)"]
    )
    if preset == "🔴 วิกฤตหมอกควันปิดเมือง (Stagnation)":
        d_temp, d_rh, d_wind, d_pm = 32.0, 85.0, 1.2, 75.0
    elif preset == "🟢 ลมพัดระบายหลังฝน (Ventilation)":
        d_temp, d_rh, d_wind, d_pm = 24.0, 45.0, 18.0, 20.0
    else:
        d_temp, d_rh, d_wind, d_pm = live_weather["temp"], live_weather["rh"], live_weather["wind"], live_air["pm25"]

    temp_val = st.sidebar.slider("อุณหภูมิ (°C)", 15.0, 45.0, float(d_temp))
    rh_val = st.sidebar.slider("ความชื้นสัมพัทธ์ RH (%)", 10.0, 100.0, float(d_rh))
    wind_val = st.sidebar.slider("ความเร็วลม (km/h)", 0.0, 30.0, float(d_wind))
    raw_input = st.sidebar.number_input("ค่าฝุ่นดิบ DustBoy กลางเมือง (µg/m³)", value=float(d_pm))

    # Stage 1 Calibration ด้วยโมเดล Random Forest (.pkl)
    if rf_model is not None:
        input_features = pd.DataFrame([{
            'pm25_raw': raw_input,
            'rh': rh_val,
            'temp': temp_val,
            'wind_speed': wind_val,
            'elevation': 240.0
        }])
        pm_base_ref = float(rf_model.predict(input_features)[0])
    else:
        # Fallback equation ถ้าหาไฟล์ .pkl ไม่เจอ
        pm_base_ref = 44.382 + (0.839 * raw_input) - (0.357 * rh_val) + (0.151 * temp_val) - (0.099 * wind_val)

# Sidebar Color Mode Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 การแสดงผลสีบนแผนที่")
color_mode = st.sidebar.radio(
    "เลือกรูปแบบเฉดสี:",
    [
        "🏥 ตามเกณฑ์มาตรฐานสุขภาพ (AQI Standard)",
        "⛰️ เน้นความต่างภูมิประเทศ (Topographical Contrast)"
    ]
)

# -------------------------------------------------------------
# 4. Stage 2 Spatial Downscaling (Topographical Gradient)
# -------------------------------------------------------------
min_elev = df_grid["elevation"].min()

# สัมประสิทธิ์ความสูง -0.04 µg/m³ ต่อเมตร (ฝุ่นสะสมก้นแอ่ง เจือจางบนยอดเขา)
df_grid["pm25_pred"] = pm_base_ref - ((df_grid["elevation"] - min_elev) * 0.04)
df_grid["pm25_pred"] = df_grid["pm25_pred"].clip(lower=2.0)

min_val = float(df_grid["pm25_pred"].min())
max_val = float(df_grid["pm25_pred"].max())

# -------------------------------------------------------------
# 5. Dashboard Metrics (KPIs)
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("ค่าฝุ่นอ้างอิงในเมือง", f"{pm_base_ref:.1f} µg/m³")
c2.metric("อุณหภูมิปัจจุบัน", f"{temp_val:.1f} °C")
c3.metric("ความชื้นสัมพัทธ์ (RH)", f"{rh_val:.1f} %")
c4.metric("ช่วงค่าฝุ่น 225 จุด", f"{min_val:.1f} - {max_val:.1f} µg/m³")

# -------------------------------------------------------------
# 6. Color Mapping Logic
# -------------------------------------------------------------
def get_marker_color(pm, mode, min_v, max_v):
    if mode == "🏥 ตามเกณฑ์มาตรฐานสุขภาพ (AQI Standard)":
        if pm >= 50.0: return "#FF0000"    # แดง (มีผลกระทบต่อสุขภาพ)
        elif pm >= 37.5: return "#FF8C00"  # ส้ม (เริ่มมีผลกระทบ)
        elif pm >= 25.0: return "#FFD700"  # เหลือง (ปานกลาง)
        elif pm >= 15.0: return "#2ECC71"  # เขียว (ดี)
        else: return "#00BFFF"             # ฟ้า (ดีมาก)
    else:
        # โหมด Relative Contrast (ยืดสเกล Min-Max เพื่อดูรอยเว้าแหว่งตามแนวเขา)
        if max_v == min_v:
            return "#00BFFF"
        ratio = (pm - min_v) / (max_v - min_v)
        if ratio >= 0.75: return "#FF4500"    # สีส้มแดง (ก้นแอ่งสะสมตัวหนาแน่น)
        elif ratio >= 0.50: return "#FFA500"  # สีส้ม (ชานเมือง)
        elif ratio >= 0.25: return "#2ECC71"  # สีเขียว (ไหล่เขา)
        else: return "#00BFFF"               # สีฟ้า (ยอดเขาสูง อากาศระบายดี)

# -------------------------------------------------------------
# 7. Folium Map Rendering (OpenStreetMap No Watermark)
# -------------------------------------------------------------
m = folium.Map(location=[18.275, 99.475], zoom_start=11, tiles="OpenStreetMap")

for _, row in df_grid.iterrows():
    pt_color = get_marker_color(row["pm25_pred"], color_mode, min_val, max_val)
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=7,
        color=pt_color,
        fill=True,
        fill_color=pt_color,
        fill_opacity=0.85,
        weight=1,
        popup=f"<b>{row['grid_id']}</b><br>PM2.5: {row['pm25_pred']:.2f} µg/m³<br>Elevation: {row['elevation']:.1f} m"
    ).add_to(m)

st_folium(m, width="100%", height=550)