"""
NaijaClimaGuard — Streamlit Dashboard v3.0 (app.py)
====================================================
New in v3:
  [1] Register / Login with session persistence
  [2] Account type vetting (individual, farmer, NGO, business, government)
  [3] Subscription tier gating — features unlocked by plan
  [4] Paystack inline payment — card tokenisation for renewals
  [5] Saved card management — charge_authorization for auto-renew
  [6] Profile & billing dashboard
  [7] 2022 Flood Replay (gated: Individual+)
  [8] Borehole Siting with monthly quota enforcement
  [9] WhatsApp alerts (gated: Individual+)

Run:
  pip install streamlit folium streamlit-folium plotly joblib pandas numpy requests bcrypt
  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import datetime
import time
import urllib.parse
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

import database as db
from database import TIERS, ACCOUNT_TYPES
from payments import (initialise_transaction, verify_transaction,
                      charge_saved_card, format_naira, PAYSTACK_PUBLIC_KEY)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NaijaClimaGuard — AI Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme toggle state ────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dm = st.session_state.dark_mode

# ── Design tokens ─────────────────────────────────────────────────────────────
if dm:
    BG       = "#0b0f1a"
    SURFACE  = "#131929"
    SURFACE2 = "#1a2236"
    BORDER   = "#243050"
    TEXT1    = "#eef2ff"
    TEXT2    = "#8899bb"
    TEXT3    = "#4a5a7a"
    ACCENT   = "#3b82f6"
    ACCENT2  = "#60a5fa"
    SUCCESS  = "#10b981"
    WARNING  = "#f59e0b"
    DANGER   = "#ef4444"
    SIDEBAR  = "#0d1525"
else:
    BG       = "#f0f4ff"
    SURFACE  = "#ffffff"
    SURFACE2 = "#e8eef8"
    BORDER   = "#c8d4ec"
    TEXT1    = "#0f1a35"
    TEXT2    = "#3a4a6a"
    TEXT3    = "#8899bb"
    ACCENT   = "#1d4ed8"
    ACCENT2  = "#2563eb"
    SUCCESS  = "#059669"
    WARNING  = "#d97706"
    DANGER   = "#dc2626"
    SIDEBAR  = "#e0e8f8"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {{
    --bg: {BG};
    --surface: {SURFACE};
    --surface2: {SURFACE2};
    --border: {BORDER};
    --text1: {TEXT1};
    --text2: {TEXT2};
    --text3: {TEXT3};
    --accent: {ACCENT};
    --accent2: {ACCENT2};
    --success: {SUCCESS};
    --warning: {WARNING};
    --danger: {DANGER};
    --sidebar: {SIDEBAR};
    --radius: 10px;
    --radius-lg: 16px;
    --shadow: 0 2px 12px rgba(0,0,0,{0.3 if dm else 0.08});
  }}

  html, body, [class*="css"], .stApp {{
    font-family: 'Sora', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text1) !important;
  }}

  [data-testid="stSidebar"] {{
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
  }}

  [data-testid="stSidebar"] * {{
    color: var(--text1) !important;
  }}

  .block-container {{ padding-top: 1.2rem !important; padding-bottom: 2rem !important; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── Metric cards with fade-in ── */
  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(12px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  .ncg-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 18px;
    animation: fadeUp 0.4s ease both;
    box-shadow: var(--shadow);
    transition: border-color 0.2s, transform 0.15s;
  }}
  .ncg-card:hover {{ border-color: var(--accent2); transform: translateY(-2px); }}

  .ncg-metric-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--accent2);
    line-height: 1;
    margin-bottom: 4px;
  }}
  .ncg-metric-label {{
    font-size: 0.72rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }}
  .ncg-metric-explain {{
    font-size: 0.78rem;
    color: var(--text3);
    line-height: 1.5;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
  }}

  /* ── Section headers ── */
  .ncg-section {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
    margin-bottom: 16px;
    margin-top: 8px;
  }}

  /* ── Risk badges ── */
  .risk-pill {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
  }}
  .risk-critical {{ background: rgba(239,68,68,0.15); color:{DANGER}; border:1px solid rgba(239,68,68,0.3); }}
  .risk-high     {{ background: rgba(245,158,11,0.15); color:{WARNING}; border:1px solid rgba(245,158,11,0.3); }}
  .risk-medium   {{ background: rgba(234,179,8,0.15); color:#eab308; border:1px solid rgba(234,179,8,0.3); }}
  .risk-low      {{ background: rgba(16,185,129,0.15); color:{SUCCESS}; border:1px solid rgba(16,185,129,0.3); }}

  /* ── Location rows ── */
  .loc-row {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
    animation: fadeUp 0.3s ease both;
  }}
  .loc-row:hover {{ border-color: var(--accent); }}

  /* ── Progress bar ── */
  .prog-track {{
    background: var(--surface2);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin: 8px 0;
  }}
  @keyframes growBar {{
    from {{ width: 0%; }}
    to   {{ width: var(--pct); }}
  }}
  .prog-fill {{
    height: 100%;
    border-radius: 4px;
    animation: growBar 0.8s cubic-bezier(0.4,0,0.2,1) both;
    width: var(--pct);
  }}

  /* ── Paywall ── */
  .ncg-paywall {{
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 40px 24px;
    text-align: center;
    margin: 16px 0;
  }}

  /* ── Tier badge pills ── */
  .tier-free       {{ background:rgba(90,106,126,0.15); color:{TEXT2}; border:1px solid {BORDER}; }}
  .tier-individual {{ background:rgba(59,130,246,0.15); color:{ACCENT2}; border:1px solid rgba(59,130,246,0.3); }}
  .tier-farmer     {{ background:rgba(16,185,129,0.15); color:{SUCCESS}; border:1px solid rgba(16,185,129,0.3); }}
  .tier-business   {{ background:rgba(245,158,11,0.15); color:{WARNING}; border:1px solid rgba(245,158,11,0.3); }}
  .tier-government {{ background:rgba(239,68,68,0.15);  color:{DANGER};  border:1px solid rgba(239,68,68,0.3); }}

  /* ── Sidebar profile card ── */
  .sidebar-profile {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    margin-bottom: 16px;
  }}

  /* ── How-to steps ── */
  .how-step {{
    display: flex;
    gap: 16px;
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
    animation: fadeUp 0.4s ease both;
  }}
  .how-step:last-child {{ border-bottom: none; }}
  .how-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent2);
    min-width: 36px;
    line-height: 1;
  }}
  .how-title {{ font-weight: 600; font-size: 0.95rem; color: var(--text1); margin-bottom: 4px; }}
  .how-body  {{ font-size: 0.83rem; color: var(--text2); line-height: 1.7; }}

  /* ── Buttons ── */
  .stButton>button {{
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: all 0.15s !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.25) !important;
  }}
  .stButton>button:hover {{
    background: var(--accent2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
  }}

  /* ── Inputs ── */
  .stTextInput>div>div>input,
  .stSelectbox>div>div,
  .stTextArea>div>div>textarea {{
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text1) !important;
    font-family: 'Sora', sans-serif !important;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: var(--surface2);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    padding: 8px 14px !important;
    background: transparent !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: var(--surface) !important;
    color: var(--accent2) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow) !important;
  }}

  /* ── Metrics (native st.metric) ── */
  div[data-testid="metric-container"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px !important;
  }}
  div[data-testid="metric-container"] label {{
    color: var(--text2) !important;
    font-size: 0.72rem !important;
  }}
  div[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: var(--accent2) !important;
    font-family: 'JetBrains Mono', monospace !important;
  }}

  /* ── Card display (saved cards) ── */
  .card-display {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text1);
  }}

  /* ── Notification box ── */
  .ncg-info {{
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: var(--radius);
    padding: 14px 18px;
    font-size: 0.84rem;
    color: var(--text2);
    line-height: 1.8;
    margin-bottom: 16px;
  }}
  .ncg-info b {{ color: var(--accent2); }}

  /* ── Paywall box ── */
  .paywall-box {{
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 36px 24px;
    text-align: center;
    margin: 16px 0;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  /* ── WhatsApp btn ── */
  .wa-btn {{
    display: block;
    background: #25D366;
    color: white !important;
    font-weight: 600;
    padding: 10px;
    border-radius: 8px;
    text-decoration: none;
    font-family: 'Sora', sans-serif;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 10px;
    transition: opacity 0.15s;
  }}
  .wa-btn:hover {{ opacity: 0.85; }}

  /* ── Header ── */
  .ncg-header {{
    background: {'linear-gradient(135deg,#0d1525 0%,#0f1e40 100%)' if dm else 'linear-gradient(135deg,#dbeafe 0%,#eff6ff 100%)'};
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }}
  .ncg-header::after {{
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
  }}
  .ncg-title {{
    font-family: 'Sora', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent2);
    margin: 0;
    letter-spacing: -0.3px;
  }}
  .ncg-subtitle {{ font-size: 0.88rem; color: var(--text2); margin-top: 4px; }}
  .ncg-live {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    color: {SUCCESS};
    font-size: 0.68rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }}
  }}
  .live-dot {{
    width: 6px; height: 6px;
    background: {SUCCESS};
    border-radius: 50%;
    animation: pulse 2s infinite;
  }}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
LOCATIONS = [
    {"name":"Lokoja",    "state":"Kogi",    "lat":7.7975,  "lon":6.7399,  "flood_site":1,"pop":300000},
    {"name":"Makurdi",   "state":"Benue",   "lat":7.7337,  "lon":8.5227,  "flood_site":1,"pop":350000},
    {"name":"Onitsha",   "state":"Anambra", "lat":6.1667,  "lon":6.7833,  "flood_site":1,"pop":560000},
    {"name":"Kano",      "state":"Kano",    "lat":12.0022, "lon":8.5920,  "flood_site":0,"pop":3800000},
    {"name":"Maiduguri", "state":"Borno",   "lat":11.8333, "lon":13.1500, "flood_site":0,"pop":700000},
]
FEATURE_COLS = [
    "precipitation_sum","precipitation_hours","temperature_2m_max","temperature_2m_min",
    "wind_speed_10m_max","et0_fao_evapotranspiration","river_discharge",
    "rain_3d_sum","rain_7d_sum","rain_14d_sum","discharge_lag1","discharge_lag3",
    "temp_range","latitude","longitude","flood_site",
]
WEATHER_API  = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_API    = "https://flood-api.open-meteo.com/v1/flood"
MODEL_PATH   = Path("flood_model.pkl")
SHAP_BAR     = Path("shap_bar.png")
SHAP_BEE     = Path("shap_beeswarm.png")
CM_IMG       = Path("confusion_matrix.png")
TRAINING_CSV = Path("training_data.csv")
OWNER_WA     = "+2348064425781"

# ── Session defaults ─────────────────────────────────────────────────────────
for k, v in {"user": None, "auth_view": "login", "payment_ref": None,
             "payment_tier": None, "show_payment": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────────────────
def risk_label(prob):
    if prob >= 0.75: return "CRITICAL", "risk-critical", DANGER
    elif prob >= 0.50: return "HIGH",   "risk-high",     WARNING
    elif prob >= 0.25: return "MEDIUM", "risk-medium",   "#eab308"
    else: return "LOW",                 "risk-low",      SUCCESS

def section(title):
    st.markdown(f'<div class="ncg-section">{title}</div>', unsafe_allow_html=True)

def info_box(html):
    st.markdown(f'<div class="ncg-info">{html}</div>', unsafe_allow_html=True)

def metric_card(value, label, explanation, color=None, delay=0):
    c = color or ACCENT2
    st.markdown(f"""
    <div class="ncg-card" style="animation-delay:{delay}s">
      <div class="ncg-metric-label">{label}</div>
      <div class="ncg-metric-val" style="color:{c}">{value}</div>
      <div class="ncg-metric-explain">{explanation}</div>
    </div>""", unsafe_allow_html=True)


def current_tier():
    if not st.session_state.user: return "free"
    return st.session_state.user.get("tier","free")

def tier_info(): return TIERS.get(current_tier(), TIERS["free"])

def has_feature(feature):
    t = tier_info()
    if feature == "replay":       return t["replay_days"] > 0
    if feature == "borehole":     return t["borehole_quota"] != 0
    if feature == "whatsapp":     return t["whatsapp_alerts"]
    if feature == "api":          return t["api_access"]
    if feature == "all_locations":return t["locations"] >= 5
    return False

def borehole_remaining():
    if not st.session_state.user: return 0
    quota = tier_info()["borehole_quota"]
    if quota == -1: return 999
    used = db.count_borehole_this_month(st.session_state.user["id"])
    return max(0, quota - used)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): return None
    return joblib.load(MODEL_PATH)

def fetch_weather(lat, lon, start, end):
    params = {"latitude":lat,"longitude":lon,"start_date":str(start),"end_date":str(end),
              "daily":"precipitation_sum,precipitation_hours,temperature_2m_max,"
                      "temperature_2m_min,wind_speed_10m_max,et0_fao_evapotranspiration",
              "timezone":"Africa/Lagos"}
    try:
        r = requests.get(WEATHER_API, params=params, timeout=25)
        r.raise_for_status()
        return r.json().get("daily",{})
    except: return {}

def fetch_discharge(lat, lon, start, end):
    params = {"latitude":lat,"longitude":lon,"start_date":str(start),
              "end_date":str(end),"daily":"river_discharge"}
    try:
        r = requests.get(FLOOD_API, params=params, timeout=25)
        r.raise_for_status()
        return r.json().get("daily",{})
    except: return {}

def build_feature_row(loc, weather, discharge):
    try:
        rain  = [float(x or 0) for x in weather.get("precipitation_sum",[0])]
        hours = [float(x or 0) for x in weather.get("precipitation_hours",[0])]
        tmax  = [float(x or 30) for x in weather.get("temperature_2m_max",[30])]
        tmin  = [float(x or 20) for x in weather.get("temperature_2m_min",[20])]
        wind  = [float(x or 10) for x in weather.get("wind_speed_10m_max",[10])]
        et0   = [float(x or 5)  for x in weather.get("et0_fao_evapotranspiration",[5])]
        disch = [float(x or 0)  for x in discharge.get("river_discharge",[0])]
        p,ph,tm,tn,w,et = rain[-1],hours[-1],tmax[-1],tmin[-1],wind[-1],et0[-1]
        d  = disch[-1]
        d1 = disch[-2] if len(disch)>=2 else d
        d3 = disch[-4] if len(disch)>=4 else d
        r3  = sum(rain[-3:]);  r7 = sum(rain[-7:]); r14 = sum(rain[-14:])
        return np.array([p,ph,tm,tn,w,et,d,r3,r7,r14,d1,d3,tm-tn,
                         loc["lat"],loc["lon"],float(loc["flood_site"])],
                        dtype=float).reshape(1,-1)
    except: return None

@st.cache_data(ttl=3600)
def get_live_predictions():
    payload = load_model()
    if not payload: return []
    model = payload["model"]
    end = datetime.date.today(); start = end - datetime.timedelta(days=16)
    results = []
    for loc in LOCATIONS:
        w = fetch_weather(loc["lat"],loc["lon"],start,end)
        d = fetch_discharge(loc["lat"],loc["lon"],start,end)
        X = build_feature_row(loc,w,d)
        prob = float(model.predict_proba(X)[0][1]) if X is not None else 0.0
        rv = w.get("precipitation_sum",[0]); dv = d.get("river_discharge",[0])
        results.append({"name":loc["name"],"state":loc["state"],"lat":loc["lat"],
                         "lon":loc["lon"],"pop":loc["pop"],"prob":prob,
                         "rain_mm":float(rv[-1]) if rv else 0.0,
                         "discharge":float(dv[-1]) if dv else 0.0})
    return results

@st.cache_data
def load_replay_data():
    if not TRAINING_CSV.exists(): return None
    df = pd.read_csv(TRAINING_CSV, parse_dates=["date"])
    return df[(df["date"]>="2022-08-01")&(df["date"]<="2022-11-30")].copy()

def get_replay_predictions(replay_df, date_str):
    payload = load_model()
    if not payload: return []
    model = payload["model"]
    day_df = replay_df[replay_df["date"]==date_str]
    results = []
    for loc in LOCATIONS:
        row = day_df[day_df["location"]==loc["name"]]
        if row.empty:
            results.append({"name":loc["name"],"state":loc["state"],"lat":loc["lat"],
                             "lon":loc["lon"],"pop":loc["pop"],"prob":0.0,
                             "discharge":0.0,"rain_mm":0.0,"flood_occurred":0})
            continue
        r = row.iloc[0]
        X = np.array([[r.get(c,0) for c in FEATURE_COLS]],dtype=float)
        try: prob = float(model.predict_proba(X)[0][1])
        except: prob = 0.0
        results.append({"name":loc["name"],"state":loc["state"],"lat":loc["lat"],
                         "lon":loc["lon"],"pop":loc["pop"],"prob":prob,
                         "discharge":float(r.get("river_discharge",0)),
                         "rain_mm":float(r.get("precipitation_sum",0)),
                         "flood_occurred":int(r.get("flood_occurred",0))})
    return results

def build_map(predictions):
    m = folium.Map(location=[9.0,8.0],zoom_start=6,tiles="CartoDB dark_matter")
    for p in predictions:
        label,_,color = risk_label(p["prob"]); pct = int(p["prob"]*100)
        if p["prob"]>=0.50:
            folium.CircleMarker(location=[p["lat"],p["lon"]],radius=28,
                color=color,fill=True,fill_color=color,fill_opacity=0.08,weight=1).add_to(m)
        popup_html = f"""<div style="font-family:'Sora',sans-serif;min-width:200px;padding:4px">
          <b style="color:{color}">{p['name']}</b>
          <span style="font-size:11px;color:#888;margin-left:6px">{p['state']}</span>
          <hr style="margin:6px 0;border-color:#333">
          <table style="width:100%;font-size:12px">
            <tr><td style="color:#888">Risk</td>
                <td style="text-align:right;color:{color};font-weight:700">{label} ({pct}%)</td></tr>
            <tr><td style="color:#888">Rain</td><td style="text-align:right">{p['rain_mm']:.1f} mm</td></tr>
            <tr><td style="color:#888">Discharge</td><td style="text-align:right">{p['discharge']:,.0f} m³/s</td></tr>
          </table></div>"""
        folium.CircleMarker(location=[p["lat"],p["lon"]],radius=14,
            color=color,fill=True,fill_color=color,fill_opacity=0.85,weight=2,
            popup=folium.Popup(popup_html,max_width=250),
            tooltip=f"{p['name']}: {label} ({pct}%)").add_to(m)
        folium.Marker(location=[p["lat"],p["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-family:monospace;font-size:10px;color:white;'
                     f'text-align:center;margin-top:18px;font-weight:700;'
                     f'text-shadow:0 0 4px #000">{pct}%</div>',
                icon_size=(40,20),icon_anchor=(20,0))).add_to(m)
    return m

def make_gauge(prob, name):
    label,_,color = risk_label(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",value=round(prob*100,1),
        number={"suffix":"%","font":{"size":30,"color":color,"family":"Space Mono"}},
        gauge={"axis":{"range":[0,100],"tickcolor":TEXT3},
               "bar":{"color":color,"thickness":0.25},"bgcolor":SURFACE,"borderwidth":0,
               "steps":[{"range":[0,25],"color":"rgba(0,200,100,0.1)"},
                         {"range":[25,50],"color":"rgba(255,215,0,0.1)"},
                         {"range":[50,75],"color":"rgba(255,140,0,0.1)"},
                         {"range":[75,100],"color":"rgba(255,59,59,0.15)"}],
               "threshold":{"line":{"color":color,"width":3},"value":prob*100}},
        title={"text":f"{name}<br><span style='font-size:11px;color:{TEXT3}'>{label}</span>",
               "font":{"size":13,"color":"#e8eaf0"}}))
    fig.update_layout(height=190,margin=dict(t=50,b=5,l=10,r=10),
                      paper_bgcolor=BG,font_color=TEXT1)
    return fig

def borehole_score(lat, lon):
    SEDIMENTARY = [(4.0,8.0,6.0,10.0),(5.0,7.5,4.0,7.5),(10.0,14.0,10.0,14.0),(7.0,9.0,4.0,7.0)]
    BASEMENT    = [(7.0,13.0,3.0,7.0),(9.0,13.0,6.0,10.0),(11.0,14.0,6.0,11.0)]
    def in_box(la,lo,b): return b[0]<=la<=b[1] and b[2]<=lo<=b[3]
    score=45; geo_type="Basement Complex"; depth="40–80m"; yld="Low–Medium (1–5 m³/hr)"
    for b in SEDIMENTARY:
        if in_box(lat,lon,b): score=78;geo_type="Sedimentary Basin";depth="15–45m";yld="High (5–20 m³/hr)";break
    for b in BASEMENT:
        if in_box(lat,lon,b): score=52;geo_type="Basement Complex";depth="35–70m";yld="Low–Medium";break
    if lat>11.5 and lon>12.0: score=88;geo_type="Chad Basin";depth="10–30m";yld="Very High (20–50 m³/hr)"
    if lat<6.0  and lon<7.5:  score=82;geo_type="Niger Delta";depth="8–25m";yld="High (10–30 m³/hr)"
    if lat<8.0: score=min(100,score+8)
    elif lat>12.0: score=max(0,score-12)
    if 6.0<=lon<=7.5 and 6.5<=lat<=9.5: score=min(100,score+10)
    if 7.5<=lon<=10.5 and 6.8<=lat<=8.5: score=min(100,score+8)
    if score>=75: verdict,vc,rec="EXCELLENT","#00c864","High confidence. Recommend drilling."
    elif score>=55: verdict,vc,rec="GOOD","#4da6ff","Good potential. Commission VES survey first."
    elif score>=35: verdict,vc,rec="MODERATE","#ffd700","Moderate. Full VES survey strongly recommended."
    else: verdict,vc,rec="LOW","#ff3b3b","Low potential. Consider rainwater harvesting."
    return {"score":score,"verdict":verdict,"verdict_color":vc,"geo_type":geo_type,
            "depth":depth,"yield":yld,"recommendation":rec}


# ════════════════════════════════════════════════════════════════════════════
# AUTH FORMS
# ════════════════════════════════════════════════════════════════════════════
def render_auth():
    st.markdown(f"""
    <div class="ncg-header">
      <div class="ncg-title">🌊 NaijaClimaGuard</div>
      <div class="ncg-subtitle">AI-Powered Flood Risk Intelligence for Nigeria</div>
    </div>""", unsafe_allow_html=True)

    col_auth, col_info = st.columns([1, 1])

    with col_auth:
        view = st.session_state.auth_view
        if view == "login":
            render_login()
        else:
            render_register()

    with col_info:
        st.markdown("""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                    padding:16px 20px;font-size:0.85rem;color:{TEXT2};line-height:1.9;
                    margin-bottom:16px">
          <b style="color:{ACCENT2}">What is NaijaClimaGuard?</b><br>
          Real-time AI flood risk intelligence for Nigeria. We fuse NASA satellite
          rainfall data and GloFAS river discharge to give daily flood risk scores —
          <b style="color:{TEXT1}">48 hours before flooding occurs.</b><br><br>
          <b style="color:{ACCENT2}">Validated on the 2022 Nigerian megaflood</b><br>
          ROC-AUC 0.9928 · 1.4M people protected · TRL-5 certified
        </div>""", unsafe_allow_html=True)

        # Demo credentials panel
        st.markdown('<div class="ncg-section">Demo Accounts</div>',
                    unsafe_allow_html=True)
        DEMO_CREDS = [
            ("free",       "🔵 Free",       "demo.free@naijaclimagard.com",       "Demo@Free1"),
            ("individual", "🟦 Individual", "demo.individual@naijaclimagard.com", "Demo@Ind1"),
            ("farmer",     "🟢 Farmer Pro", "demo.farmer@naijaclimagard.com",     "Demo@Farm1"),
            ("business",   "🟡 Business",   "demo.business@naijaclimagard.com",   "Demo@Biz1"),
            ("government", "🟠 Government", "demo.government@naijaclimagard.com", "Demo@Gov1"),
        ]
        TIER_COLORS_DEMO = {
            "free":"#5a6a7e","individual":"#4da6ff",
            "farmer":"#00c864","business":"#ffd700","government":"#ff8c00"
        }
        for tier_key, label, email, password in DEMO_CREDS:
            color = TIER_COLORS_DEMO[tier_key]
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};
                        border-left:3px solid {color};
                        border-radius:8px;padding:10px 14px;
                        margin-bottom:7px;font-size:0.8rem">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:4px">
                <span style="color:{color};font-weight:600">{label}</span>
              </div>
              <div style="color:{TEXT2};font-family:'JetBrains Mono',monospace;
                          font-size:0.72rem">
                📧 {email}<br>
                🔑 {password}
              </div>
            </div>""", unsafe_allow_html=True)

        # One-click demo login buttons
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ncg-section">One-Click Demo Login</div>',
                    unsafe_allow_html=True)
        demo_cols = st.columns(5)
        for col, (tier_key, label, email, password) in zip(demo_cols, DEMO_CREDS):
            color = TIER_COLORS_DEMO[tier_key]
            with col:
                short = label.split()[-1]
                if st.button(short, key=f"demo_login_{tier_key}"):
                    user, err = db.authenticate_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error(err)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="border-top:1px solid {BORDER};padding-top:24px"></div>',
                unsafe_allow_html=True)
    render_pricing_page(show_signup_buttons=True)


def render_login():
    st.markdown('<div class="ncg-section">Sign In</div>', unsafe_allow_html=True)
    with st.form("login_form"):
        email    = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In →")
        if submitted:
            user, err = db.authenticate_user(email, password)
            if err:
                st.error(err)
            else:
                st.session_state.user = user
                db.audit(user["id"], "login")
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Create a free account →"):
        st.session_state.auth_view = "register"
        st.rerun()


def render_register():
    st.markdown('<div class="ncg-section">Create Account</div>', unsafe_allow_html=True)
    with st.form("register_form"):
        full_name    = st.text_input("Full name *")
        email        = st.text_input("Email address *")
        phone        = st.text_input("WhatsApp number (e.g. +2348012345678) *")
        password     = st.text_input("Password (min 8 characters) *", type="password")
        password2    = st.text_input("Confirm password *", type="password")
        account_type = st.selectbox("Account type *",
                                    ["individual","farmer","ngo","business",
                                     "government","researcher"],
                                    format_func=lambda x: {
                                        "individual":"Individual / Personal",
                                        "farmer":"Farmer / Agricultural",
                                        "ngo":"NGO / Humanitarian",
                                        "business":"Business / Corporate",
                                        "government":"Government / Public Agency",
                                        "researcher":"Academic / Researcher",
                                    }[x])
        state        = st.selectbox("State *", [
            "Abia","Adamawa","Akwa Ibom","Anambra","Bauchi","Bayelsa","Benue",
            "Borno","Cross River","Delta","Ebonyi","Edo","Ekiti","Enugu","FCT",
            "Gombe","Imo","Jigawa","Kaduna","Kano","Katsina","Kebbi","Kogi",
            "Kwara","Lagos","Nasarawa","Niger","Ogun","Ondo","Osun","Oyo",
            "Plateau","Rivers","Sokoto","Taraba","Yobe","Zamfara"])
        organisation = st.text_input("Organisation / Company (optional)")
        submitted = st.form_submit_button("Create Account →")
        if submitted:
            if not all([full_name, email, phone, password]):
                st.error("All fields marked * are required.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = db.create_user(email, password, full_name, phone,
                                          account_type, state, organisation)
                if ok:
                    user, _ = db.authenticate_user(email, password)
                    st.session_state.user = user
                    db.audit(user["id"], "register", account_type)
                    st.success("Account created! Welcome to NaijaClimaGuard.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to sign in"):
        st.session_state.auth_view = "login"
        st.rerun()


def render_tier_overview():
    render_pricing_page(show_signup_buttons=True)


def render_pricing_page(show_signup_buttons=False):
    TIER_COLORS = {
        "free":       "#5a6a7e",
        "individual": "#4da6ff",
        "farmer":     "#00c864",
        "business":   "#ffd700",
        "government": "#ff8c00",
    }

    # Feature matrix rows
    FEATURES = [
        ("Live risk map",              {"free":"1 location","individual":"5 locations","farmer":"5 locations","business":"5 locations","government":"5 locations"}),
        ("Flood risk gauges",          {"free":"✅","individual":"✅","farmer":"✅","business":"✅","government":"✅"}),
        ("2022 Flood Replay",          {"free":"🔒","individual":"7 days","farmer":"30 days","business":"Full archive","government":"Full archive"}),
        ("Borehole siting reports",    {"free":"🔒","individual":"3/month","farmer":"10/month","business":"Unlimited","government":"Unlimited"}),
        ("WhatsApp flood alerts",      {"free":"🔒","individual":"✅","farmer":"✅","business":"✅","government":"✅"}),
        ("Planting calendar AI",       {"free":"🔒","individual":"🔒","farmer":"✅","business":"✅","government":"✅"}),
        ("Historical data access",     {"free":"🔒","individual":"7 days","farmer":"30 days","business":"5 years","government":"5 years"}),
        ("API access",                 {"free":"🔒","individual":"🔒","farmer":"🔒","business":"✅","government":"✅"}),
        ("Custom LGA reports",         {"free":"🔒","individual":"🔒","farmer":"🔒","business":"🔒","government":"✅"}),
        ("SLA support",                {"free":"🔒","individual":"🔒","farmer":"🔒","business":"Email","government":"Dedicated"}),
    ]

    st.markdown('<div class="ncg-section">Plans & Pricing</div>', unsafe_allow_html=True)

    # Tier cards
    cols = st.columns(len(TIERS))
    for col, (tier_key, t) in zip(cols, TIERS.items()):
        color = TIER_COLORS.get(tier_key, "#4da6ff")
        price_str = "Free" if t["price_monthly"] == 0 else f"₦{t['price_monthly']:,}"
        period    = "" if t["price_monthly"] == 0 else "/mo"
        is_popular = tier_key == "individual"
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE};border:{'2px' if is_popular else '1px'} solid
                        {color if is_popular else '{BORDER}'};
                        border-radius:12px;padding:18px 14px;text-align:center;
                        position:relative;margin-bottom:4px">
              {'<div style="position:absolute;top:-10px;left:50%;transform:translateX(-50%);'
               'background:'+color+';color:#0a0e1a;font-family:JetBrains Mono,monospace;'
               'font-size:0.6rem;font-weight:700;padding:2px 10px;border-radius:20px;'
               'white-space:nowrap">MOST POPULAR</div>' if is_popular else ''}
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                          color:{color};font-weight:700;letter-spacing:1px;
                          margin-bottom:6px">{t['badge']}</div>
              <div style="font-size:1.5rem;font-weight:700;color:{TEXT1};
                          font-family:'JetBrains Mono',monospace">{price_str}
                <span style="font-size:0.75rem;color:{TEXT3}">{period}</span>
              </div>
              <div style="font-size:0.72rem;color:{TEXT3};margin-top:4px">
                {t['label']}
              </div>
            </div>""", unsafe_allow_html=True)
            if show_signup_buttons and tier_key != "free":
                if st.button(f"Get {t['label']}",
                             key=f"pricing_signup_{tier_key}"):
                    st.session_state.auth_view = "register"
                    st.rerun()

    # Feature comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ncg-section">Feature Comparison</div>',
                unsafe_allow_html=True)

    tier_keys = list(TIERS.keys())
    tier_labels = [TIERS[k]["label"] for k in tier_keys]
    header_colors = [TIER_COLORS.get(k, "#4da6ff") for k in tier_keys]

    # Build table HTML
    header_cells = "".join([
        f'<th style="text-align:center;padding:8px 6px;'
        f'color:{c};font-family:JetBrains Mono,monospace;'
        f'font-size:0.68rem;font-weight:700;letter-spacing:1px;'
        f'border-bottom:2px solid {c}">{l}</th>'
        for c, l in zip(header_colors, tier_labels)
    ])

    rows_html = ""
    for feat_name, feat_vals in FEATURES:
        cells = "".join([
            f'<td style="text-align:center;padding:9px 6px;'
            f'font-size:0.78rem;color:{"#5a6a7e" if feat_vals[k]=="🔒" else "#e8eaf0"};'
            f'border-bottom:1px solid {BORDER}">{feat_vals[k]}</td>'
            for k in tier_keys
        ])
        rows_html += f"""<tr>
          <td style="padding:9px 10px;font-size:0.8rem;color:{TEXT2};
                     border-bottom:1px solid {BORDER};white-space:nowrap">{feat_name}</td>
          {cells}
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;background:{SURFACE};
                  border-radius:10px;overflow:hidden">
      <thead>
        <tr>
          <th style="text-align:left;padding:10px;color:{TEXT3};
                     font-size:0.72rem;border-bottom:1px solid {BORDER}">Feature</th>
          {header_cells}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    <br>
    <div style="font-size:0.75rem;color:{TEXT3};text-align:center">
      All plans include: Open-Meteo ERA5 satellite data · GloFAS river discharge ·
      AI flood prediction · Mobile-friendly dashboard
    </div>""", unsafe_allow_html=True)

    # dummy closing for old function — this block intentionally empty
    if False:
        st.markdown(f"""
        <div></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAYMENT FLOW
# ════════════════════════════════════════════════════════════════════════════
def render_payment_modal(tier_key):
    t = TIERS[tier_key]
    user = st.session_state.user
    saved_card = db.get_default_card(user["id"])

    st.markdown(f"""
    <div style="background:{SURFACE};border:2px solid {BORDER};border-radius:14px;
                padding:28px;max-width:500px;margin:0 auto;text-align:center">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                  color:{ACCENT2};margin-bottom:8px">UPGRADE PLAN</div>
      <div style="font-size:1.4rem;font-weight:700;margin-bottom:4px">{t['label']}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
                  color:{ACCENT2};margin-bottom:16px">
        {format_naira(t['price_kobo'])}<span style="font-size:0.9rem;color:{TEXT3}">/month</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if saved_card:
        st.markdown(f"""
        <div class="card-display">
          <div style="font-size:0.7rem;color:{TEXT3};margin-bottom:6px;
                      text-transform:uppercase;letter-spacing:1px">Saved Card</div>
          <div style="font-size:1rem;letter-spacing:3px;color:{TEXT1}">
            •••• •••• •••• {saved_card['last4']}</div>
          <div style="font-size:0.78rem;color:{TEXT2};margin-top:4px">
            {saved_card['card_type'].upper()} · Expires {saved_card['exp_month']}/{saved_card['exp_year']}
            · {saved_card['bank']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Pay {format_naira(t['price_kobo'])} with saved card"):
                result = charge_saved_card(
                    user["email"], t["price_kobo"],
                    saved_card["authorization_code"], tier_key, user["id"]
                )
                if result["status"]:
                    db.create_subscription(user["id"], tier_key,
                                           result["reference"], t["price_kobo"])
                    db.audit(user["id"], "payment_saved_card",
                             f"{tier_key}:{result['reference']}")
                    st.session_state.user = db.get_user(user["id"])
                    st.session_state.show_payment = False
                    st.success(f"✅ Upgraded to {t['label']}!")
                    time.sleep(1); st.rerun()
                else:
                    st.error(f"Payment failed: {result['error']}")
        with col2:
            if st.button("Use a different card"):
                st.session_state.pop("_use_new_card", None)
                st.session_state["_use_new_card"] = True
                st.rerun()

    if not saved_card or st.session_state.get("_use_new_card"):
        st.markdown('<div class="ncg-section">Pay with Card</div>', unsafe_allow_html=True)
        # Inline Paystack popup via JS
        ref = f"NCG-{user['id']}-{tier_key}-{int(time.time())}"
        paystack_js = f"""
        <script src="https://js.paystack.co/v1/inline.js"></script>
        <button onclick="payWithPaystack()" style="
          background:#4da6ff;color:#0a0e1a;border:none;border-radius:8px;
          font-family:'JetBrains Mono',monospace;font-weight:700;font-size:0.82rem;
          padding:12px 28px;cursor:pointer;width:100%;margin-top:8px">
          💳 Pay {format_naira(t['price_kobo'])} Securely
        </button>
        <div id="pay-status" style="margin-top:12px;font-size:0.82rem;color:{TEXT2}"></div>
        <script>
        function payWithPaystack() {{
          var handler = PaystackPop.setup({{
            key:       '{PAYSTACK_PUBLIC_KEY}',
            email:     '{user["email"]}',
            amount:    {t['price_kobo']},
            currency:  'NGN',
            ref:       '{ref}',
            metadata:  {{ user_id: {user["id"]}, tier: '{tier_key}' }},
            onClose: function() {{
              document.getElementById('pay-status').innerText = 'Payment window closed.';
            }},
            callback: function(response) {{
              document.getElementById('pay-status').innerText =
                '✅ Payment received! Reference: ' + response.reference +
                '. Please click Verify Payment below.';
              document.getElementById('pay-ref').value = response.reference;
              document.getElementById('verify-btn').style.display = 'block';
            }}
          }});
          handler.openIframe();
        }}
        </script>
        <input type="hidden" id="pay-ref" value="">
        <button id="verify-btn"
          style="display:none;background:#00c864;color:#0a0e1a;border:none;
                 border-radius:8px;font-family:'JetBrains Mono',monospace;font-weight:700;
                 font-size:0.82rem;padding:10px 24px;cursor:pointer;width:100%;margin-top:10px"
          onclick="verifyPayment()">
          ✅ Verify Payment
        </button>
        <script>
        function verifyPayment() {{
          var ref = document.getElementById('pay-ref').value;
          if (ref) {{
            window.location.href = window.location.pathname +
              '?payment_ref=' + ref + '&payment_tier={tier_key}';
          }}
        }}
        </script>
        """
        components.html(paystack_js, height=160)

    if st.button("✕  Cancel"):
        st.session_state.show_payment = False
        st.rerun()


def handle_payment_callback():
    """Check URL params for Paystack callback after redirect."""
    try:
        params = dict(st.query_params)
    except Exception:
        params = st.experimental_get_query_params()
    _ref  = params.get("payment_ref")
    _tier = params.get("payment_tier")
    # experimental_get_query_params returns lists; query_params returns strings
    ref  = _ref[0]  if isinstance(_ref,  list) else _ref
    tier = _tier[0] if isinstance(_tier, list) else _tier
    if not ref or not tier: return

    user = st.session_state.user
    if not user: return

    result = verify_transaction(ref)
    if result["status"]:
        db.create_subscription(user["id"], tier, ref, result["amount"])
        if result.get("reusable") and result.get("authorization_code"):
            db.save_card(user["id"],
                         result["authorization_code"],
                         result.get("card_type","unknown"),
                         result.get("last4","0000"),
                         result.get("exp_month","00"),
                         result.get("exp_year","00"),
                         result.get("bank","unknown"))
        db.audit(user["id"], "payment_verified", f"{tier}:{ref}")
        st.session_state.user = db.get_user(user["id"])
        try: st.query_params.clear()
        except Exception: st.experimental_set_query_params()
        st.success(f"✅ Payment verified! You now have {TIERS[tier]['label']} access.")
        time.sleep(1); st.rerun()
    else:
        st.error(f"Payment verification failed: {result['error']}")
        try: st.query_params.clear()
        except Exception: st.experimental_set_query_params()


def paywall(feature_name, min_tier):
    t = TIERS[min_tier]
    # Use feature_name hash for a unique key so multiple paywalls on the same page don't clash
    btn_key = f"paywall_{min_tier}_{abs(hash(feature_name)) % 99999}"
    st.markdown(f"""
    <div class="paywall-box">
      <div style="font-size:2rem;margin-bottom:12px">🔒</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                  font-weight:700;color:{TEXT1};margin-bottom:8px">
        {feature_name}
      </div>
      <div style="font-size:0.85rem;color:{TEXT2};margin-bottom:20px">
        Available from <b style="color:{ACCENT2}">{t['label']}</b> —
        {format_naira(t['price_kobo'])}/month
      </div>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button(f"Upgrade to {t['label']} →", key=btn_key):
            st.session_state.show_payment = True
            st.session_state.payment_tier = min_tier
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    user = st.session_state.user
    tier = current_tier()
    t    = tier_info()
    TIER_C = {"free":TEXT3,"individual":ACCENT2,"farmer":SUCCESS,
               "business":WARNING,"government":DANGER}
    color  = TIER_C.get(tier, ACCENT2)

    with st.sidebar:
        col_logo, col_theme = st.columns([3,1])
        with col_logo:
            st.markdown(f"""
            <div style="padding:12px 0 8px">
              <div style="font-family:'Sora',sans-serif;font-size:1.1rem;
                          font-weight:700;color:{ACCENT2}">NaijaClimaGuard</div>
              <div style="font-size:0.67rem;color:{TEXT3};letter-spacing:1.5px;
                          text-transform:uppercase;margin-top:2px">AI Flood Intelligence</div>
            </div>""", unsafe_allow_html=True)
        with col_theme:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("☀️" if dm else "🌙", key="theme_toggle",
                         help="Switch light/dark mode"):
                st.session_state.dark_mode = not dm
                st.rerun()

        st.markdown(f'<div style="height:1px;background:{BORDER};margin:4px 0 14px"></div>',
                    unsafe_allow_html=True)

        if user:
            st.markdown(f"""
            <div class="sidebar-profile">
              <div style="font-weight:600;font-size:0.9rem;color:{TEXT1}">{user['full_name']}</div>
              <div style="font-size:0.73rem;color:{TEXT3};margin-top:2px;margin-bottom:8px">
                {user['email']}</div>
              <span class="risk-pill" style="background:rgba(59,130,246,0.1);
                    color:{color};border:1px solid {color};font-size:0.63rem">
                {t['badge']}
              </span>
              <span style="font-size:0.72rem;color:{TEXT3};margin-left:6px">
                {user['account_type'].title()}</span>
            </div>""", unsafe_allow_html=True)

            if t["borehole_quota"] > 0:
                remaining = borehole_remaining()
                used_n = t["borehole_quota"] - remaining
                pct_used = int(used_n / t["borehole_quota"] * 100)
                st.markdown(f"""
                <div style="margin-bottom:14px">
                  <div style="font-size:0.72rem;color:{TEXT3};margin-bottom:5px">
                    Borehole reports: <b style="color:{TEXT2}">{remaining} left</b> of {t['borehole_quota']}
                  </div>
                  <div class="prog-track">
                    <div class="prog-fill"
                         style="--pct:{pct_used}%;
                                background:{WARNING if pct_used>70 else SUCCESS}"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            upgradeable = {k:v for k,v in TIERS.items()
                           if v["price_monthly"] > t["price_monthly"]}
            if upgradeable:
                st.markdown(f'<div style="font-size:0.68rem;color:{TEXT3};'
                            f'text-transform:uppercase;letter-spacing:1.5px;'
                            f'margin-bottom:8px">Upgrade Plan</div>', unsafe_allow_html=True)
                for uk, uv in list(upgradeable.items())[:3]:
                    if st.button(f"{uv['label']} — {format_naira(uv['price_kobo'])}/mo",
                                 key=f"sidebar_upgrade_{uk}"):
                        st.session_state.show_payment = True
                        st.session_state.payment_tier = uk
                        st.rerun()

            st.markdown(f'<div style="height:1px;background:{BORDER};margin:14px 0"></div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-size:0.68rem;color:{TEXT3};text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:10px">AI Model Stats</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px">
              <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:8px;
                          padding:10px;text-align:center">
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                            font-weight:600;color:{ACCENT2}">0.9928</div>
                <div style="font-size:0.63rem;color:{TEXT3};margin-top:2px">Accuracy Score</div>
              </div>
              <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:8px;
                          padding:10px;text-align:center">
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                            font-weight:600;color:{SUCCESS}">91%</div>
                <div style="font-size:0.63rem;color:{TEXT3};margin-top:2px">Precision</div>
              </div>
            </div>""", unsafe_allow_html=True)

            col_r, col_s = st.columns(2)
            with col_r:
                if st.button("Refresh", key="refresh_btn"):
                    st.cache_data.clear(); st.rerun()
            with col_s:
                if st.button("Sign Out", key="signout_btn"):
                    st.session_state.user = None
                    st.session_state.auth_view = "login"
                    st.rerun()
        else:
            st.markdown(f"""
            <div style="font-size:0.83rem;color:{TEXT3};padding:8px 0;line-height:1.9">
              Sign in or create a free account to access NaijaClimaGuard.
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP — execution starts here
# ════════════════════════════════════════════════════════════════════════════

# Step 1: Render sidebar (always safe regardless of auth state)
render_sidebar()

# Step 2: Auth gate
if not st.session_state.user:
    render_auth()
    st.stop()

# Step 3: Logged-in user flow
user = st.session_state.user
handle_payment_callback()

# Step 4: Payment modal
if st.session_state.get("show_payment") and st.session_state.get("payment_tier"):
    st.markdown("""
    <div class="ncg-header">
      <div class="ncg-title">🌊 NaijaClimaGuard</div>
      <div class="ncg-subtitle">Upgrade your subscription</div>
    </div>""", unsafe_allow_html=True)
    render_payment_modal(st.session_state.payment_tier)
    st.stop()

# Step 5: Model check
if not MODEL_PATH.exists():
    st.error("flood_model.pkl not found. Run model_trainer.py first.")
    st.stop()

# Header
tier = current_tier(); t = tier_info()
tier_colors = {"free":"#5a6a7e","individual":"#4da6ff",
               "farmer":"#00c864","business":"#ffd700","government":"#ff8c00"}
tier_color = tier_colors.get(tier,"#4da6ff")

st.markdown(f"""
<div class="ncg-header">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div class="ncg-title">🌊 NaijaClimaGuard</div>
      <div class="ncg-subtitle">AI-Powered Flood Risk Intelligence — Nigeria's 774 LGAs</div>
      <div class="ncg-live"><div class="live-dot"></div> LIVE &nbsp;·&nbsp; {datetime.date.today().strftime("%d %B %Y")}</div>
    </div>
    <div style="text-align:right">
      <span style="background:rgba(77,166,255,0.1);border:1px solid {tier_color};
                   color:{tier_color};font-family:'JetBrains Mono',monospace;
                   font-size:0.72rem;padding:4px 12px;border-radius:20px">
        {t['badge']}
      </span>
      <div style="font-size:0.72rem;color:{TEXT3};margin-top:4px">
        {user['full_name']} · {user['account_type'].title()}
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# Tabs
tab_live, tab_replay, tab_borehole, tab_alerts, tab_billing, tab_model, tab_pricing, tab_howto = st.tabs([
    "Live Risk Map",
    "2022 Flood Replay",
    "Borehole Siting",
    "WhatsApp Alerts",
    "Billing & Cards",
    "Model Validation",
    "Plans & Pricing",
    "How To Use",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE RISK MAP
# ════════════════════════════════════════════════════════════════════════════
with tab_live:
    with st.spinner("Fetching live data..."):
        all_preds = get_live_predictions()

    allowed_n = t["locations"]
    predictions = all_preds[:allowed_n]

    if not has_feature("all_locations"):
        st.info(f"🔒 Free plan shows 1 location. Upgrade to see all 5.")

    max_risk  = max(p["prob"] for p in predictions)
    avg_rain  = np.mean([p["rain_mm"] for p in predictions])
    max_disch = max(p["discharge"] for p in predictions)
    total_pop = sum(p["pop"] for p in predictions)

    lbl, cls, risk_color = risk_label(max_risk)
    rain_status = "Heavy — flooding possible" if avg_rain > 30 else ("Moderate" if avg_rain > 10 else "Light — conditions normal")
    disch_status = "Above danger threshold" if max_disch > 16442 else "Within safe range"

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        metric_card(
            f"{int(max_risk*100)}%",
            "Highest Flood Risk Today",
            f"The location with the worst conditions right now is rated <b style='color:{risk_color}'>{lbl}</b>. "
            f"This means our AI model sees a {int(max_risk*100)}% chance of flooding there today.",
            color=risk_color, delay=0
        )
    with c2:
        metric_card(
            f"{total_pop/1e6:.1f}M",
            "People We Are Watching Over",
            f"Our system currently monitors {total_pop/1e6:.1f} million Nigerians "
            f"across {len(predictions)} locations. When risk rises, these communities get warned first.",
            delay=0.1
        )
    with c3:
        metric_card(
            f"{avg_rain:.1f} mm",
            "Average Rainfall Today",
            f"{rain_status}. Rainfall above 30mm/day saturates soil and raises flood risk significantly. "
            f"1mm of rain = 1 litre of water per square metre of land.",
            color=WARNING if avg_rain > 30 else ACCENT2, delay=0.2
        )
    with c4:
        metric_card(
            f"{max_disch:,.0f} m³/s",
            "River Flow Speed (Peak)",
            f"{disch_status}. m³/s means cubic metres of water per second passing a point. "
            f"The Niger at Lokoja becomes dangerous above 16,442 m³/s — our AI-proven danger line.",
            color=DANGER if max_disch > 16442 else ACCENT2, delay=0.3
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_map, col_g = st.columns([3,2])
    with col_map:
        section("Real-Time Risk Map — Click any dot for details")
        st_folium(build_map(predictions), width=None, height=450, returned_objects=[])
    with col_g:
        section("Location Risk Gauges")
        info_box("Each gauge shows the AI's confidence that flooding will occur today "
                 "at that location. <b>Above 50% = take precautions. Above 75% = evacuate low-lying areas.</b>")
        for p in sorted(predictions, key=lambda x: x["prob"], reverse=True)[:3]:
            st.plotly_chart(make_gauge(p["prob"],p["name"]),
                            use_container_width=True,config={"displayModeBar":False})

    st.markdown('<div class="ncg-section" style="margin-top:18px">All Locations</div>',
                unsafe_allow_html=True)
    for p in sorted(predictions, key=lambda x: x["prob"], reverse=True):
        label,cls,color = risk_label(p["prob"]); pct=int(p["prob"]*100)
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                    padding:12px 16px;margin-bottom:7px">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <div><span style="font-weight:600">{p['name']}</span>
                 <span style="color:{TEXT3};font-size:0.75rem;margin-left:7px">
                 {p['state']} · {p['pop']:,} people</span></div>
            <span class="{cls}" style="font-family:'JetBrains Mono',monospace;font-size:0.8rem">
              {label} · {pct}%</span>
          </div>
          <div style="background:{SURFACE2};border-radius:4px;height:5px">
            <div style="background:{color};width:{max(pct,1)}%;height:100%;border-radius:4px"></div>
          </div>
          <div style="display:flex;gap:18px;margin-top:6px;font-size:0.75rem;color:{TEXT3}">
            <span>🌧 {p['rain_mm']:.1f} mm</span><span>🌊 {p['discharge']:,.0f} m³/s</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — 2022 REPLAY (gated: Individual+)
# ════════════════════════════════════════════════════════════════════════════
with tab_replay:
    if not has_feature("replay"):
        paywall("2022 Flood Replay — Watch the AI detect Nigeria's worst flood in real time",
                "individual")
    else:
        replay_df = load_replay_data()
        if replay_df is None:
            st.error("training_data.csv not found.")
        else:
            st.markdown("""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                        padding:14px 18px;margin-bottom:18px;font-size:0.85rem;
                        color:{TEXT2};line-height:1.8">
              <b style="color:{ACCENT2}">2022 Nigerian Megaflood Replay:</b>
              Drag the slider to watch the AI flag flood risk day by day.
              The model flagged Lokoja as CRITICAL on October 1 —
              48 hours before official government advisories.
            </div>""", unsafe_allow_html=True)

            all_dates = sorted(replay_df["date"].dt.date.unique())
            selected_date = st.select_slider(
                "Travel through the 2022 flood:",
                options=all_dates, value=all_dates[0])

            date_str = str(selected_date)
            replay_preds = get_replay_predictions(replay_df, date_str)
            day_max = max((p["prob"] for p in replay_preds), default=0)
            day_disch = max((p["discharge"] for p in replay_preds), default=0)
            flood_confirmed = sum(1 for p in replay_preds if p["flood_occurred"])
            lbl,cls,_ = risk_label(day_max)

            rd1,rd2,rd3,rd4 = st.columns(4)
            with rd1:
                st.markdown(f"""<div class="ncg-card">
                  <div class="metric-val {cls}">{int(day_max*100)}%</div>
                  <div class="ncg-metric-label">Peak Risk</div></div>""",unsafe_allow_html=True)
            with rd2:
                st.markdown(f"""<div class="ncg-card">
                  <div class="ncg-metric-val">{day_disch:,.0f}</div>
                  <div class="ncg-metric-label">Peak Discharge m³/s</div></div>""",unsafe_allow_html=True)
            with rd3:
                st.markdown(f"""<div class="ncg-card">
                  <div class="ncg-metric-val">{flood_confirmed}</div>
                  <div class="ncg-metric-label">Flood Confirmed</div></div>""",unsafe_allow_html=True)
            with rd4:
                st.markdown(f"""<div class="ncg-card">
                  <div class="ncg-metric-val">{selected_date.strftime('%d %b')}</div>
                  <div class="ncg-metric-label">2022</div></div>""",unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            rc1,rc2 = st.columns([3,2])
            with rc1:
                st.markdown('<div class="ncg-section">AI Risk Map</div>',unsafe_allow_html=True)
                st_folium(build_map(replay_preds), width=None, height=400,
                          returned_objects=[], key=f"r_{date_str}")
            with rc2:
                st.markdown('<div class="ncg-section">Lokoja Discharge</div>',unsafe_allow_html=True)
                lok = replay_df[replay_df["location"]=="Lokoja"].sort_values("date")
                if not lok.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=lok["date"],y=lok["river_discharge"],
                        mode="lines",line=dict(color="#4da6ff",width=2),
                        fill="tozeroy",fillcolor="rgba(77,166,255,0.1)",name="Discharge"))
                    fd = lok[lok["flood_occurred"]==1]
                    if not fd.empty:
                        fig.add_trace(go.Scatter(x=fd["date"],y=fd["river_discharge"],
                            mode="markers",marker=dict(color="#ff3b3b",size=5),
                            name="Flood Day"))
                    vts = int(pd.Timestamp(date_str).timestamp()*1000)
                    fig.add_vline(x=vts,line_dash="dash",line_color="#ffd700",
                                  line_width=2,annotation_text="← Now",
                                  annotation_font_color="#ffd700",annotation_font_size=10)
                    fig.add_hline(y=16442.76,line_dash="dot",line_color="#ff8c00",
                                  line_width=1,annotation_text="90th pct",
                                  annotation_font_color="#ff8c00",annotation_font_size=9)
                    fig.update_layout(height=380,paper_bgcolor=BG,
                        plot_bgcolor=SURFACE,margin=dict(t=15,b=25,l=10,r=10),
                        font=dict(color=TEXT2,size=10),
                        xaxis=dict(gridcolor=BORDER),
                        yaxis=dict(gridcolor=BORDER,title="m³/s"),
                        legend=dict(font=dict(color=TEXT2,size=9),bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

            for p in sorted(replay_preds, key=lambda x: x["prob"], reverse=True):
                label,cls,color = risk_label(p["prob"])
                fc = "🔴 CONFIRMED" if p["flood_occurred"] else "🟢 No flood"
                st.markdown(f"""
                <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
                            padding:11px 15px;margin-bottom:6px;display:flex;
                            justify-content:space-between;align-items:center">
                  <span style="font-weight:600">{p['name']}</span>
                  <span style="font-size:0.73rem;color:{TEXT3}">{fc}</span>
                  <span style="font-size:0.73rem;color:{TEXT3}">
                    🌧{p['rain_mm']:.0f}mm 🌊{p['discharge']:,.0f}m³/s</span>
                  <span class="{cls}" style="font-family:'JetBrains Mono',monospace;font-size:0.78rem">
                    {label}·{int(p['prob']*100)}%</span>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BOREHOLE (gated: Individual+, quota enforced)
# ════════════════════════════════════════════════════════════════════════════
with tab_borehole:
    if not has_feature("borehole"):
        paywall("Borehole Siting Intelligence — AI groundwater scoring for any location in Nigeria",
                "individual")
    else:
        remaining = borehole_remaining()
        quota = t["borehole_quota"]

        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                    padding:14px 18px;margin-bottom:18px;font-size:0.85rem;
                    color:{TEXT2};line-height:1.8">
          <b style="color:{ACCENT2}">Borehole pre-screening:</b>
          Enter any GPS coordinates. Get geological zone, estimated depth, yield,
          and drilling recommendation. Professional surveys cost ₦2–5M.
          This costs {'unlimited reports — Business plan' if quota==-1
                      else f'{remaining} of {quota} reports remaining this month'}.
        </div>""", unsafe_allow_html=True)

        if remaining == 0:
            paywall("You've used all borehole reports this month — upgrade for more","business")
        else:
            bc1,bc2 = st.columns(2)
            with bc1:
                presets = {"Custom":None,"Lokoja,Kogi":(7.7975,6.7399),
                           "Abuja,FCT":(9.0579,7.4951),"Lagos":(6.455,3.384),
                           "Port Harcourt":(4.816,7.050),"Kano":(12.002,8.592),
                           "Maiduguri":(11.833,13.150),"Ibadan":(7.378,3.947)}
                preset = st.selectbox("Quick-select city:",list(presets.keys()))
                dlat,dlon = presets[preset] if presets[preset] else (9.0579,7.4951)
                b_lat = st.number_input("Latitude", value=dlat, format="%.4f", step=0.001)
                b_lon = st.number_input("Longitude",value=dlon, format="%.4f", step=0.001)
                run_b = st.button("🔬  Analyse Groundwater Potential")

            with bc2:
                result = borehole_score(b_lat, b_lon)
                if run_b:
                    db.log_borehole(user["id"],b_lat,b_lon,
                                    result["score"],result["verdict"])
                    db.audit(user["id"],"borehole_report",
                             f"{b_lat},{b_lon}:{result['verdict']}")
                    st.session_state.user = db.get_user(user["id"])

                sc = result["score"]; vc = result["verdict_color"]
                fig_s = go.Figure(go.Indicator(
                    mode="gauge+number",value=sc,
                    number={"suffix":"/100","font":{"size":32,"color":vc,"family":"Space Mono"}},
                    gauge={"axis":{"range":[0,100]},"bar":{"color":vc,"thickness":0.28},
                           "bgcolor":SURFACE,"borderwidth":0,
                           "steps":[{"range":[0,35],"color":"rgba(255,59,59,0.1)"},
                                     {"range":[35,55],"color":"rgba(255,215,0,0.1)"},
                                     {"range":[55,75],"color":"rgba(77,166,255,0.1)"},
                                     {"range":[75,100],"color":"rgba(0,200,100,0.1)"}]},
                    title={"text":f"<b style='color:{vc}'>{result['verdict']}</b>",
                           "font":{"size":16,"color":"#e8eaf0"}}))
                fig_s.update_layout(height=210,margin=dict(t=55,b=5,l=10,r=10),
                                    paper_bgcolor=BG)
                st.plotly_chart(fig_s,use_container_width=True,config={"displayModeBar":False})
                st.markdown(f"""
                <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                            padding:16px;font-size:0.82rem">
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                    <div><div style="color:{TEXT3};font-size:0.68rem;text-transform:uppercase">
                           Geological Zone</div>
                         <div style="color:{TEXT1};font-weight:600">{result['geo_type']}</div></div>
                    <div><div style="color:{TEXT3};font-size:0.68rem;text-transform:uppercase">
                           Estimated Depth</div>
                         <div style="color:{TEXT1};font-weight:600">{result['depth']}</div></div>
                    <div><div style="color:{TEXT3};font-size:0.68rem;text-transform:uppercase">
                           Expected Yield</div>
                         <div style="color:{TEXT1};font-weight:600">{result['yield']}</div></div>
                    <div><div style="color:{TEXT3};font-size:0.68rem;text-transform:uppercase">
                           Reports Left</div>
                         <div style="color:#00c864;font-weight:600">
                           {borehole_remaining() if quota!=-1 else '∞'}</div></div>
                  </div>
                  <div style="margin-top:12px;padding-top:10px;border-top:1px solid {BORDER};
                              color:{TEXT2};line-height:1.7">
                    <b style="color:{ACCENT2}">Recommendation:</b> {result['recommendation']}
                  </div>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — WHATSAPP ALERTS (gated: Individual+)
# ════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    if not has_feature("whatsapp"):
        paywall("WhatsApp Flood Alerts — Daily risk notifications for your location","individual")
    else:
        st.markdown('<div class="ncg-section">Subscribe to Flood Alerts</div>',
                    unsafe_allow_html=True)
        predictions_for_alerts = get_live_predictions()
        phone = user.get("phone","")

        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid #25D366;border-radius:10px;
                    padding:16px 20px;margin-bottom:20px;font-size:0.85rem;
                    color:{TEXT2};line-height:1.9">
          Alerts will be sent to your registered WhatsApp: <b style="color:{TEXT1}">{phone}</b><br>
          Click below to open a pre-filled WhatsApp message to activate your alert for each location.
        </div>""", unsafe_allow_html=True)

        al_cols = st.columns(len(predictions_for_alerts))
        for col, p in zip(al_cols, predictions_for_alerts):
            label,cls,color = risk_label(p["prob"])
            msg = (f"NAIJACLIMAGARD ALERT\n\nName: {user['full_name']}\n"
                   f"Location: {p['name']}, {p['state']}\n"
                   f"Account: {user['account_type'].title()}\n"
                   f"Current Risk: {label} ({int(p['prob']*100)}%)\n\n"
                   f"Reply START to activate daily flood alerts.")
            wa_url = f"https://wa.me/{OWNER_WA.replace('+','')}?text={urllib.parse.quote(msg)}"
            with col:
                st.markdown(f"""
                <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                            padding:14px;text-align:center">
                  <div style="font-weight:600;margin-bottom:3px">{p['name']}</div>
                  <div style="font-size:0.73rem;color:{TEXT3};margin-bottom:9px">
                    {p['state']} State</div>
                  <div class="{cls}" style="font-family:'JetBrains Mono',monospace;
                               font-size:0.78rem;margin-bottom:11px">
                    {label}·{int(p['prob']*100)}%</div>
                  <a href="{wa_url}" target="_blank" style="
                    display:block;background:#25D366;color:white;
                    font-weight:700;padding:10px;border-radius:8px;
                    text-decoration:none;font-family:'JetBrains Mono',monospace;
                    font-size:0.75rem;">💬 Activate Alert</a>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — BILLING & CARDS
# ════════════════════════════════════════════════════════════════════════════
with tab_billing:
    st.markdown('<div class="ncg-section">Subscription & Payment</div>',unsafe_allow_html=True)

    sub = db.get_active_subscription(user["id"])
    cards = db.get_all_cards(user["id"])

    bill1, bill2 = st.columns(2)

    with bill1:
        st.markdown('<div class="ncg-section">Current Plan</div>',unsafe_allow_html=True)
        tier_color_map = {"free":"#5a6a7e","individual":"#4da6ff","farmer":"#00c864",
                          "business":"#ffd700","government":"#ff8c00"}
        tc = tier_color_map.get(tier,"#4da6ff")
        st.markdown(f"""
        <div style="background:{SURFACE};border:2px solid {tc};border-radius:12px;
                    padding:20px;text-align:center;margin-bottom:16px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                      color:{TEXT3};margin-bottom:6px">ACTIVE PLAN</div>
          <div style="font-size:1.4rem;font-weight:700;color:{tc}">{t['label']}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                      margin-top:6px;color:{TEXT1}">
            {format_naira(t['price_kobo']) if t['price_kobo']>0 else 'Free'}/month
          </div>
          {'<div style="font-size:0.78rem;color:{TEXT3};margin-top:8px">Expires: '
           + sub["expires_at"][:10] + '</div>' if sub else ''}
        </div>""", unsafe_allow_html=True)

        # Feature checklist
        features = [
            (f"{t['locations']} location{'s' if t['locations']>1 else ''}","✅"),
            ("2022 Flood Replay","✅" if t['replay_days']>0 else "🔒"),
            ("Borehole Reports",
             f"✅ {t['borehole_quota']}/mo" if t['borehole_quota']>0
             else ("✅ Unlimited" if t['borehole_quota']==-1 else "🔒")),
            ("WhatsApp Alerts","✅" if t['whatsapp_alerts'] else "🔒"),
            ("API Access","✅" if t['api_access'] else "🔒"),
        ]
        for feat, status in features:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:7px 0;
                        border-bottom:1px solid {BORDER};font-size:0.83rem">
              <span style="color:{TEXT2}">{feat}</span>
              <span>{status}</span>
            </div>""", unsafe_allow_html=True)

        if tier != "government":
            st.markdown("<br>", unsafe_allow_html=True)
            upgrade_options = {k:v for k,v in TIERS.items()
                               if v["price_monthly"] > t["price_monthly"]}
            for uk,uv in list(upgrade_options.items())[:2]:
                if st.button(f"Upgrade to {uv['label']} — {format_naira(uv['price_kobo'])}/mo",
                             key=f"billing_upgrade_{uk}"):
                    st.session_state.show_payment = True
                    st.session_state.payment_tier = uk
                    st.rerun()

    with bill2:
        st.markdown('<div class="ncg-section">Saved Cards</div>',unsafe_allow_html=True)
        if cards:
            for card in cards:
                is_default = card["is_default"]
                st.markdown(f"""
                <div class="card-display" style="margin-bottom:10px;
                  {'border-color:#00c864;' if is_default else ''}">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <div style="font-size:0.95rem;letter-spacing:2px;color:{TEXT1}">
                        •••• •••• •••• {card['last4']}</div>
                      <div style="font-size:0.73rem;color:{TEXT3};margin-top:3px">
                        {card['card_type'].upper()} · {card['exp_month']}/{card['exp_year']}
                        · {card['bank']}</div>
                    </div>
                    {'<span style="font-size:0.65rem;color:#00c864;border:1px solid #00c864;'
                     'padding:2px 7px;border-radius:10px">DEFAULT</span>' if is_default else ''}
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:{SURFACE};border:1px dashed {BORDER};border-radius:10px;
                        padding:24px;text-align:center;color:{TEXT3};font-size:0.85rem">
              No saved cards yet.<br>Cards are saved automatically after your first payment
              for seamless monthly renewal.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ncg-section">Payment History</div>',unsafe_allow_html=True)
        conn = db.get_conn()
        history = conn.execute(
            """SELECT tier, amount_paid, paystack_ref, started_at, expires_at
               FROM subscriptions WHERE user_id=? ORDER BY created_at DESC LIMIT 10""",
            (user["id"],)
        ).fetchall()
        conn.close()
        if history:
            for h in history:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid {BORDER};font-size:0.78rem">
                  <span style="color:{TEXT2}">{h['started_at'][:10]}</span>
                  <span style="color:{TEXT1}">{TIERS.get(h['tier'],{}).get('label',h['tier'])}</span>
                  <span style="color:#00c864">
                    {format_naira(h['amount_paid']) if h['amount_paid'] else '—'}</span>
                  <span style="color:{TEXT3};font-family:'JetBrains Mono',monospace;font-size:0.68rem">
                    {(h['paystack_ref'] or '')[:18]}...</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:{TEXT3};font-size:0.82rem'>No payment history yet.</div>",
                        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL VALIDATION
# ════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="ncg-section">TRL-5 Validation Artifacts</div>',unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    for col,(label,val,sub) in zip([m1,m2,m3,m4,m5],[
        ("ROC-AUC","0.9928","Holdout"),("Flood F1","0.81","Holdout"),
        ("Precision","91%","Flood class"),("Recall","73%","Flood class"),
        ("CV ROC-AUC","0.983","5-fold")]):
        with col:
            st.markdown(f"""<div class="ncg-card">
              <div class="ncg-metric-val" style="font-size:1.4rem">{val}</div>
              <div class="ncg-metric-label">{label}</div>
              <div style="font-size:0.65rem;color:{TEXT3};margin-top:2px">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sc1,sc2 = st.columns(2)
    with sc1:
        st.markdown('<div class="ncg-section">SHAP Feature Importance</div>',unsafe_allow_html=True)
        if SHAP_BAR.exists(): st.image(str(SHAP_BAR),use_container_width=True)
    with sc2:
        st.markdown('<div class="ncg-section">Confusion Matrix</div>',unsafe_allow_html=True)
        if CM_IMG.exists(): st.image(str(CM_IMG),use_container_width=True)
    if SHAP_BEE.exists():
        st.markdown('<div class="ncg-section" style="margin-top:16px">SHAP Beeswarm</div>',
                    unsafe_allow_html=True)
        st.image(str(SHAP_BEE),use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — PLANS & PRICING
# ════════════════════════════════════════════════════════════════════════════
with tab_pricing:
    current_plan = current_tier()
    t_current = tier_info()

    st.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:14px 20px;margin-bottom:20px;font-size:0.86rem;
                color:{TEXT2};line-height:1.8">
      You are currently on the <b style="color:{ACCENT2}">{t_current['label']}</b> plan.
      Upgrade anytime — your new plan activates immediately and your card is saved
      for seamless monthly renewal.
    </div>""", unsafe_allow_html=True)

    render_pricing_page(show_signup_buttons=False)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ncg-section">Upgrade Now</div>', unsafe_allow_html=True)

    upgrade_cols = st.columns(len(TIERS) - 1)
    upgradeable = {k: v for k, v in TIERS.items()
                   if v["price_monthly"] > t_current["price_monthly"]}
    TIER_COLORS_UP = {"individual":"#4da6ff","farmer":"#00c864",
                      "business":"#ffd700","government":"#ff8c00"}

    for col, (uk, uv) in zip(upgrade_cols, upgradeable.items()):
        color = TIER_COLORS_UP.get(uk, "#4da6ff")
        with col:
            st.markdown(f"""
            <div style="text-align:center;margin-bottom:8px">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                          color:{color};font-weight:700">{uv['label']}</div>
              <div style="font-size:1.1rem;font-weight:700;color:{TEXT1};
                          font-family:'JetBrains Mono',monospace">
                ₦{uv['price_monthly']:,}<span style="font-size:0.7rem;color:{TEXT3}">/mo</span>
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Upgrade →", key=f"pricing_tab_upgrade_{uk}"):
                st.session_state.show_payment = True
                st.session_state.payment_tier = uk
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — HOW TO USE
# ════════════════════════════════════════════════════════════════════════════
with tab_howto:
    section("How NaijaClimaGuard Works")

    info_box(
        "NaijaClimaGuard is an AI system that monitors rainfall, river levels, and soil "
        "conditions every day — then tells you, in plain language, how likely flooding is "
        "for your area. No technical knowledge needed. Here is everything explained."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Step by step guide
    STEPS = [
        ("01", "The AI reads satellite data every day",
         "Each morning, our system automatically pulls the latest weather data from NASA "
         "satellites and the Global Flood Awareness System (GloFAS). This includes how much "
         "rain fell, how fast rivers are flowing, and how wet the soil already is. "
         "You don't need to do anything — this happens automatically."),

        ("02", "The model calculates a flood risk score",
         "Our trained AI model looks at 16 different signals — including 7-day rainfall "
         "accumulation, river discharge speed, and soil saturation — and produces a single "
         "percentage: the probability that flooding will occur at each location today. "
         "A score above 50% means conditions are dangerous. Above 75% means critical."),

        ("03", "You see it on the Live Risk Map",
         "Open the Live Risk Map tab. Each coloured dot on the map is a monitored location. "
         "Green = safe. Yellow = watch closely. Orange = prepare. Red = danger. "
         "Click any dot to see the exact rainfall, river speed, and risk score for that location."),

        ("04", "Understand what each number means",
         "Flood Risk % — how confident the AI is that flooding will happen today. "
         "Rainfall (mm) — how many millimetres of rain fell. 30mm in one day is heavy rain. "
         "River Discharge (m³/s) — how many cubic metres of water flow past a point per second. "
         "Above 16,442 m³/s at Lokoja, the Niger becomes dangerous. "
         "Population Monitored — how many people live in the areas we are watching."),

        ("05", "Watch the 2022 Flood Replay to understand the scale",
         "The 2022 floods were Nigeria's worst in a decade — 1.4 million people displaced. "
         "In the Flood Replay tab, drag the date slider from August to October 2022 and watch "
         "our AI correctly flag Lokoja as CRITICAL on October 1st — 48 hours before the "
         "government issued advisories. This is proof the system works."),

        ("06", "Use Borehole Siting before you drill",
         "Before spending ₦500,000–5 million on a borehole, use our Borehole Siting tool. "
         "Enter your GPS coordinates and we will tell you: the geological zone under your land, "
         "estimated depth to water (e.g. 15–45m), expected water yield, and whether drilling "
         "is likely to succeed. This alone can save you millions in failed boreholes."),

        ("07", "Subscribe to WhatsApp alerts",
         "If you are on an Individual plan or higher, activate WhatsApp alerts for your location. "
         "When our AI detects rising risk, it sends a message to your WhatsApp number — "
         "giving you time to move livestock, vehicles, and valuables before the flood arrives. "
         "No app to download. Just WhatsApp, which you already have."),

        ("08", "Upgrade your plan as your needs grow",
         "Start free — you get the live risk map for one location. "
         "Upgrade to Individual (₦1,000/mo) to see all 5 locations, access the flood replay, "
         "and get WhatsApp alerts. Farmer Pro adds a planting calendar. "
         "Business adds API access for your own systems. Government adds custom LGA dashboards. "
         "You can upgrade or cancel anytime from the Billing tab."),
    ]

    for num, title, body in STEPS:
        st.markdown(f"""
        <div class="how-step" style="animation-delay:{int(num)*0.05}s">
          <div class="how-num">{num}</div>
          <div>
            <div class="how-title">{title}</div>
            <div class="how-body">{body}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Understanding the Model Validation Tab")

    faq = [
        ("What is ROC-AUC 0.9928?",
         "This is our model's accuracy score. It ranges from 0.5 (random guessing) to 1.0 (perfect). "
         "0.9928 means our AI is right 99.28% of the time when distinguishing flood days from "
         "safe days. This was tested on the real 2022 flood data — not lab conditions."),
        ("What does Flood F1 Score 0.81 mean?",
         "F1 balances two things: how often we correctly warn about real floods (Recall: 73%), "
         "and how often our warnings are genuine (Precision: 91%). An F1 of 0.81 means when "
         "we raise an alarm, it is almost always real — and we catch 73 out of every 100 actual flood events."),
        ("What is the SHAP chart?",
         "SHAP shows which pieces of data the AI relies on most. River discharge being at the top "
         "means the AI primarily watches river speed to detect floods — which is exactly correct "
         "hydrology. This proves the model learned real flood science, not random patterns."),
        ("What is the Confusion Matrix?",
         "It shows how the AI performed on the 2022 test data: "
         "1,823 days correctly identified as safe, 104 flood days correctly caught, "
         "38 flood days missed, and only 10 false alarms. "
         "In real-world early warning, false alarms are far cheaper than missed floods."),
    ]

    for q, a in faq:
        with st.expander(q):
            st.markdown(f"""
            <div style="font-size:0.86rem;color:{TEXT2};line-height:1.8;padding:4px 0">
              {a}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Frequently Asked Questions")

    faqs2 = [
        ("Is the data truly live?",
         "Yes. Every time you open the app or click Refresh, it fetches the latest available "
         "data from Open-Meteo (atmospheric) and GloFAS (river discharge). In April the readings "
         "will show LOW risk because it is dry season — that is correct, not a bug. "
         "Risk climbs from June through November, Nigeria's flood season."),
        ("Why only 5 locations?",
         "We started with the 5 most flood-prone and data-rich locations for our NBTI NextGen 2026 "
         "submission. Scaling to all 774 LGAs is Phase 5 of our roadmap. Every location just "
         "needs a latitude and longitude — the pipeline handles the rest automatically."),
        ("Is my payment information safe?",
         "Yes. We never store your card number. When you pay, Paystack processes your card "
         "and sends us only an authorization_code — a token that lets us charge your saved card "
         "for renewals without ever seeing the card details. This is the same system used by "
         "Uber, Spotify, and major Nigerian fintechs."),
        ("Can I get a refund?",
         "Contact us via WhatsApp at +2348064425781 within 48 hours of payment. "
         "We review every request fairly."),
    ]

    for q, a in faqs2:
        with st.expander(q):
            st.markdown(f"""
            <div style="font-size:0.86rem;color:{TEXT2};line-height:1.8;padding:4px 0">
              {a}
            </div>""", unsafe_allow_html=True)
