import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import string
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #2c3e50; color: white; }
    .metric-card {
        background-color: white; border: 1px solid #e0e0e0; 
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stMetric { background-color: white; border-radius: 10px; padding: 10px; border: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA ENGINE (Your Updated Loader) ---
@st.cache_data(ttl=300)
def load_data_v3():
    if "gcp_service_account" not in st.secrets:
        st.error("⚠️ Secrets Missing: Please add Google Credentials.")
        return None

    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)

    def fetch_sheet(sheet_id, range_name, header_row=0):
        try:
            result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = result.get('values', [])
            if not vals: return pd.DataFrame()
            
            if len(vals) > header_row:
                headers = vals[header_row]
                data = vals[header_row+1:]
                clean_headers = []
                seen = {}
                for h in headers:
                    h = str(h).strip()
                    if h in seen:
                        seen[h] += 1
                        clean_headers.append(f"{h}_{seen[h]}")
                    else:
                        seen[h] = 0
                        clean_headers.append(h)
                
                max_len = len(clean_headers)
                padded_data = [row + [None]*(max_len-len(row)) for row in data]
                return pd.DataFrame(padded_data, columns=clean_headers)
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"⚠️ Load Error ({range_name}): {str(e)}")
            return pd.DataFrame()

    IDS = {
        'cars': "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg",
        'orders': "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A",
        'clients': "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg",
        'expenses': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_expenses': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'collections': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg"
    }

    with st.spinner("🔄 Syncing HQ Data..."):
        dfs = {}
        # Fetching A:ZZ to get all columns
        dfs['cars'] = fetch_sheet(IDS['cars'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['orders'] = fetch_sheet(IDS['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 1) # Headers on Row 2
        dfs['clients'] = fetch_sheet(IDS['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        return dfs

# --- HELPER FUNCTIONS ---
def get_col_by_letter(df, letter):
    def letter_to_index(col_str):
        num = 0
        for c in col_str:
            if c.upper() in string.ascii_uppercase:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num - 1
    idx = letter_to_index(letter)
    if idx < len(df.columns): return df.columns[idx]
    return None

def clean_id_tag(x):
    if pd.isna(x): return ""
    return str(x).strip().replace(" ", "").lower()

# --- 4. NAVIGATION & INIT ---
st.sidebar.title("🚘 Rental OS 3.0")
page = st.sidebar.radio("Navigate", ["🏠 Operations", "🚗 Vehicle 360", "👥 CRM", "💰 Financial HQ", "⚠️ Risk Radar"])
st.sidebar.markdown("---")

dfs = load_data_v3()

# --- 5. MODULE 1: OPERATIONS ---
def show_operations(dfs):
    st.title("🏠 Operations Command Center")
    
    if dfs:
        df_orders = dfs['orders']
        df_cars = dfs['cars']
        
        # --- A. PROCESS DATA ---
        # 1. Active Fleet Count (Column AZ in Cars)
        col_status = get_col_by_letter(df_cars, 'AZ')
        active_fleet_count = 0
        car_map = {} # Code -> Name
        
        if col_status:
            active_cars = df_cars[df_cars[col_status].astype(str).str.contains('Valid|ساري', case=False, na=False)]
            active_fleet_count = len(active_cars)
            
            # Build Car Map: Code (A) -> Name (B+E+H+I) / Plate (AC..W)
            for _, row in active_cars.iterrows():
                try:
                    c_code = clean_id_tag(row.iloc[0]) # Col A
                    c_name = f"{row.iloc[1]} {row.iloc[4]} ({row.iloc[7]})" # B, E, H
                    c_plate = f"{row.iloc[28]} {row.iloc[27]} {row.iloc[26]}".strip() # AC, AB, AA (Partial plate)
                    car_map[c_code] = f"{c_name} | {c_plate}"
                except: continue

        # 2. Process Orders (Start: L, End: V)
        today = datetime.now()
        active_rentals = 0
        returning_today = 0
        departing_today = 0
        timeline_data = []
        
        col_start = get_col_by_letter(df_orders, 'L')
        col_end = get_col_by_letter(df_orders, 'V')
        col_car = get_col_by_letter(df_orders, 'C')
        col_client = get_col_by_letter(df_orders, 'B')
        
        if col_start and col_end:
            for _, row in df_orders.iterrows():
                try:
                    # Parse Dates
                    s_date = pd.to_datetime(row[col_start], errors='coerce')
                    e_date = pd.to_datetime(row[col_end], errors='coerce')
                    
                    if pd.isna(s_date) or pd.isna(e_date): continue
                    
                    # Live Metrics Logic
                    if s_date <= today <= e_date: active_rentals += 1
                    if e_date.date() == today.date(): returning_today += 1
                    if s_date.date() == today.date(): departing_today += 1
                    
                    # Timeline Data
                    car_code = clean_id_tag(row[col_car])
                    car_label = car_map.get(car_code, f"Unknown ({car_code})")
                    
                    timeline_data.append({
                        'Car': car_label,
                        'Start': s_date,
                        'End': e_date,
                        'Client': str(row[col_client]),
                        'Status': 'Active' if s_date <= today <= e_date else 'Scheduled'
                    })
                except: continue

        fleet_utilization = int((active_rentals / active_fleet_count * 100)) if active_fleet_count > 0 else 0

        # --- B. UI VISUALS ---
        # 1. Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚗 Active Rentals", active_rentals, delta="Live on Road")
        c2.metric("🔄 Returning Today", returning_today, delta_color="inverse")
        c3.metric("🛫 Departing Today", departing_today)
        c4.metric("📊 Fleet Utilization", f"{fleet_utilization}%", f"{active_fleet_count} Total Cars")
        
        st.divider()
        
        # 2. The Huge Calendar
        st.subheader("📅 Live Fleet Schedule")
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Gantt Chart
            fig = px.timeline(
                df_timeline, 
                x_start="Start", 
                x_end="End", 
                y="Car", 
                color="Status",
                hover_data=["Client"],
                color_discrete_map={"Active": "#2ecc71", "Scheduled": "#3498db"}
            )
            fig.update_yaxes(autorange="reversed") # Cars top to bottom
            fig.update_layout(
                height=600, 
                xaxis_title="Date",
                font=dict(size=14),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            
            # Highlight Today
            fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active bookings found in the system.")

# --- 6. PAGE ROUTER ---
if page == "🏠 Operations": show_operations(dfs)
elif page == "🚗 Vehicle 360": st.title("🚗 Vehicle 360 (Coming Next)")
elif page == "👥 CRM": st.title("👥 CRM (Coming Next)")
elif page == "💰 Financial HQ": st.title("💰 Financial HQ (Coming Next)")
elif page == "⚠️ Risk Radar": st.title("⚠️ Risk Radar (Coming Next)")
