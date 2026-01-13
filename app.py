import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘")

# --- 2. CUSTOM CSS (The Professional Look) ---
st.markdown("""
<style>
    /* Main Layout */
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #f8f9fa; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #2c3e50; color: white; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white; }
    
    /* Metrics */
    .metric-card {
        background-color: white; border: 1px solid #e0e0e0; 
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; }
    
    /* Status Tags */
    .tag-green { background: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .tag-red { background: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .tag-orange { background: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 3. NAVIGATION ---
st.sidebar.title("🚘 Rental OS 3.0")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Operations (العمليات)", 
    "🚗 Vehicle 360 (السيارات)", 
    "👥 CRM (العملاء)", 
    "💰 Financial HQ (المالية)", 
    "⚠️ Risk Radar (المخاطر)"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status: Online 🟢")

# --- 4. PLACEHOLDER PAGES ---

def show_operations():
    st.title("🏠 Operations Command Center")
    st.markdown("### The 'Now' View")
    
    # Placeholder Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Rentals", "0", "Live")
    c2.metric("Returning Today", "0", "Action Req")
    c3.metric("Departing Today", "0")
    c4.metric("Fleet Available", "0%")
    
    st.divider()
    st.subheader("📅 Live Schedule (Gantt Chart)")
    st.info("ℹ️ Connect 'Orders' Sheet to visualize bookings here.")

def show_vehicle_360():
    st.title("🚗 Vehicle 360° Profile")
    st.markdown("### Granular Car Performance")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.selectbox("Select Vehicle", ["Loading Fleet..."])
        st.image("https://img.icons8.com/color/96/car--v1.png", width=100)
    
    with col2:
        st.info("ℹ️ Connect 'Cars' Sheet to see P&L per car.")

def show_crm():
    st.title("👥 Client Relationship Management")
    st.markdown("### Client Profiles & History")
    st.info("ℹ️ Connect 'Clients' & 'Orders' to build profiles.")

def show_financial_hq():
    st.title("💰 Financial Headquarters")
    
    # The 3-Bucket System Tabs
    tab1, tab2, tab3 = st.tabs(["🌊 Cash Flow (Liquidity)", "📉 Profit & Loss (Real)", "🤝 Owner Payouts"])
    
    with tab1:
        st.subheader("Liquidity Position")
        st.write("Real Cash in Hand (Income + Deposits + Prepayments)")
        st.warning("Needs: 'Collections' Sheet mapping.")
    
    with tab2:
        st.subheader("True Income Statement")
        st.write("Realized Earnings (Excluding Deposits & Future Prepayments)")
    
    with tab3:
        st.subheader("Owner Ledger")
        st.write("Net Payouts Calculation")

def show_risk_radar():
    st.title("⚠️ Risk Management Radar")
    st.markdown("### Expiries & Maintenance")
    st.error("Data Source Disconnected: Licenses, Insurance, Oil Changes.")

# --- 5. PAGE ROUTER ---
if page == "🏠 Operations (العمليات)": show_operations()
elif page == "🚗 Vehicle 360 (السيارات)": show_vehicle_360()
elif page == "👥 CRM (العملاء)": show_crm()
elif page == "💰 Financial HQ (المالية)": show_financial_hq()
elif page == "⚠️ Risk Radar (المخاطر)": show_risk_radar()

# --- 6. DATA ENGINE (The Heart of OS 3.0) ---
@st.cache_data(ttl=300) # Auto-refresh every 5 mins
def load_data_v3():
    if "gcp_service_account" not in st.secrets:
        st.error("⚠️ Secrets Missing: Please add Google Credentials.")
        return None

    # Connect to Google Sheets
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)

    def fetch_sheet(sheet_id, range_name, header_row=0):
        try:
            result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = result.get('values', [])
            if not vals: return pd.DataFrame()
            
            # Handle headers
            if len(vals) > header_row:
                headers = vals[header_row]
                data = vals[header_row+1:]
                
                # Deduplicate headers
                seen = {}
                clean_headers = []
                for h in headers:
                    h = str(h).strip()
                    if h in seen:
                        seen[h] += 1
                        clean_headers.append(f"{h}_{seen[h]}")
                    else:
                        seen[h] = 0
                        clean_headers.append(h)
                
                # Pad rows to match header length
                max_len = len(clean_headers)
                padded_data = [row + [None]*(max_len-len(row)) for row in data]
                
                return pd.DataFrame(padded_data, columns=clean_headers)
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"⚠️ Load Error ({range_name}): {str(e)}")
            return pd.DataFrame()

    # --- SHEET IDs (Your Provided IDs) ---
    IDS = {
        'cars': "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg",
        'orders': "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A",
        'clients': "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg",
        'expenses': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_expenses': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'collections': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg"
    }

    # --- FETCH DATA ---
    with st.spinner("🔄 Syncing with HQ..."):
        dfs = {}
        # Fetching specific ranges to ensure we get all columns (A:ZZ)
        dfs['cars'] = fetch_sheet(IDS['cars'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['orders'] = fetch_sheet(IDS['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 1) # Headers on Row 2 (Index 1)
        dfs['clients'] = fetch_sheet(IDS['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['expenses'] = fetch_sheet(IDS['expenses'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['car_expenses'] = fetch_sheet(IDS['car_expenses'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['collections'] = fetch_sheet(IDS['collections'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)

    return dfs

# --- HELPER: Column Index Finder ---
def get_col_by_letter(df, letter):
    """Converts Excel Letter (e.g. 'AE') to DataFrame Column Name"""
    def letter_to_index(col_str):
        num = 0
        for c in col_str:
            if c.upper() in string.ascii_uppercase:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num - 1
    
    idx = letter_to_index(letter)
    if idx < len(df.columns):
        return df.columns[idx]
    return None

# --- HELPER: Advanced Cleaner ---
def clean_currency(x):
    if pd.isna(x): return 0.0
    s = str(x).replace(',', '').replace('%', '').strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

def clean_id_tag(x):
    """Standardizes IDs for matching (removes spaces, lowercase)"""
    if pd.isna(x): return ""
    return str(x).strip().replace(" ", "").lower()
