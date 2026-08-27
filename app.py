import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import joblib
from streamlit_folium import st_folium

# -------------------------------------------------------------
# 1. การตั้งค่าหน้าเว็บ (Page Configuration)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Lampang PM2.5 Spatial Monitoring",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ ระบบแผนที่ติดตาม PM2.5 เชิงพื้นที่ 225 จุด (อ.เมืองลำปาง)")
st.caption("ระบบจำลองและคาดการณ์ตามโครงสร้างภูมิประเทศแอ่งกระทะ (Two-Stage Spatial Machine Learning Framework)")

# -------------------------------------------------------------
# 2. ฟังก์ชันโหลดโมเดล Machine Learning ทั้ง 2 ขั้นตอน
# -------------------------------------------------------------
@st.cache_resource
def load_ml_models():
    """โหลดโมเดล Random Forest ของทั้ง Stage 1 และ Stage 2"""
    try:
        stage1 = joblib.load("rf_stage1_model.pkl")
        stage2 = joblib.load("rf_stage2_spatial_model.pkl")
        return stage1, stage2
    except Exception as e:
        st.error(f"⚠️ ไม่พบไฟล์โมเดล .pkl กรุณาตรวจสอบว่ามี rf_stage1_model.pkl และ rf_stage2_spatial_model.pkl อยู่ในโฟลเดอร์: {e}")
        return None, None

rf_stage1, rf_stage2 = load_ml_models()

# -------------------------------------------------------------
# 3. ฟังก์ชันดึงข้อมูล API และตารางกริด (With Caching)
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_live_air4thai(station_code="35t"):
    """ดึงค่าฝุ่นจริงรายชั่วโมงจากกรมควบคุมมลพิษ (สถานี 35t พระบาท)"""
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
    """โหลดตารางกริดพิกัด 225 จุดพร้อมความสูงจริง (DEM Elevation)"""
    try:
        return pd.read_csv("grid_lampang.csv")
    except Exception:
        # Fallback กรณีไม่มีไฟล์กริด
        lats = np.linspace(18.20, 18.35, 15)
        lons = np.linspace(99.40, 99.55, 15)
        records = []
        for i, (lat, lon) in enumerate(zip(np.repeat(lats, 15), np.tile(lons, 15))):
            records.append({"grid_id": f"GRID_{i+1:03d}", "lat": lat, "lon": lon, "elevation": 240.0})
        return pd.DataFrame(records)

live_air = fetch_live_air4thai("35t")
live_weather = fetch_live_weather()
df_grid = load_grid_data().copy()

# -------------------------------------------------------------
# 4. แผงควบคุม Sidebar (UI & Inputs)
# -------------------------------------------------------------
st.sidebar.header("🎛️ แผงควบคุมและสถานการณ์")

app_mode = st.sidebar.radio(
    "เลือกโหมดการทำงาน:",
    ["📡 แสดงผลข้อมูลสด (Real-Time Live)", "🧪 จำลองสถานการณ์ (What-If Simulator)"]
)

if app_mode == "📡 แสดงผลข้อมูลสด (Real-Time Live)":
    st.sidebar.success(f"เชื่อมต่อ: {live_air['status']}")
    st.sidebar.caption(f"📍 {live_air['station_name']}\n\n🕒 {live_air['datetime']}")
    
    # ดึงค่าจริงจาก Air4Thai และ Open-Meteo
    pm_calibrated_base = float(live_air["pm25"])
    temp_val = float(live_weather["temp"])
    rh_val = float(live_weather["rh"])
    wind_val = float(live_weather["wind"])
    
    st.sidebar.info(f"🌫️ PM2.5 อ้างอิง: {pm_calibrated_base:.1f} µg/m³\n\n🌡️ Temp: {temp_val:.1f} °C\n\n💧 RH: {rh_val:.1f} %\n\n💨 Wind: {wind_val:.1f} km/h")

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

    # Stage 1: นำค่าฝุ่นดิบเข้าโมเดล Random Forest เพื่อตัดผลกระทบความชื้น
    if rf_stage1 is not None:
        input_s1 = pd.DataFrame([{
            'pm25_raw': raw_input,
            'rh': rh_val,
            'temp': temp_val,
            'wind_speed': wind_val,
            'elevation': 240.0
        }])
        pm_calibrated_base = float(rf_stage1.predict(input_s1)[0])
    else:
        pm_calibrated_base = raw_input

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
# 5. การรันโมเดล Stage 2: Spatial Random Forest (Pure ML)
# -------------------------------------------------------------
if rf_stage2 is not None:
    # ประกอบชุดตัวแปรต้น 225 จุด (Coordinate-Free: ตัด lat, lon ออก)
    spatial_features = pd.DataFrame({
        'pm25_calibrated': [pm_calibrated_base] * len(df_grid),
        'elevation': df_grid['elevation'],
        'temp': [temp_val] * len(df_grid),
        'rh': [rh_val] * len(df_grid),
        'wind_speed': [wind_val] * len(df_grid)
    })
    
    # ให้ Random Forest Stage 2 ทำนายค่าฝุ่นรายกริดตามความสูงจริง
    df_grid["pm25_pred"] = rf_stage2.predict(spatial_features)
else:
    df_grid["pm25_pred"] = pm_calibrated_base

min_val = float(df_grid["pm25_pred"].min())
max_val = float(df_grid["pm25_pred"].max())

# -------------------------------------------------------------
# 6. แสดงผลตัวเลขสถิติ (KPIs)
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("ค่าฝุ่นฐานกลางเมือง", f"{pm_calibrated_base:.1f} µg/m³")
c2.metric("อุณหภูมิที่ใช้", f"{temp_val:.1f} °C")
c3.metric("ความชื้นสัมพัทธ์ (RH)", f"{rh_val:.1f} %")
c4.metric("ช่วงค่าฝุ่นบนกริด 225 จุด", f"{min_val:.1f} - {max_val:.1f} µg/m³")

# -------------------------------------------------------------
# 7. ฟังก์ชันกำหนดสี (Color Mapping Logic)
# -------------------------------------------------------------
def get_marker_color(pm, mode, min_v, max_v):
    if mode == "🏥 ตามเกณฑ์มาตรฐานสุขภาพ (AQI Standard)":
        if pm >= 50.0: return "#FF0000"    # สีแดง
        elif pm >= 37.5: return "#FF8C00"  # สีส้ม
        elif pm >= 25.0: return "#FFD700"  # สีเหลือง
        elif pm >= 15.0: return "#2ECC71"  # สีเขียว
        else: return "#00BFFF"             # สีฟ้า
    else:
        if max_v == min_v:
            return "#00BFFF"
        ratio = (pm - min_v) / (max_v - min_v)
        if ratio >= 0.75: return "#FF4500"    # ก้นแอ่งกระทะ
        elif ratio >= 0.50: return "#FFA500"  # ชานเมือง
        elif ratio >= 0.25: return "#2ECC71"  # ไหล่เขา
        else: return "#00BFFF"               # ยอดเขาสูง

# -------------------------------------------------------------
# 8. แสดงผลแผนที่ Folium (OpenStreetMap No Watermark)
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