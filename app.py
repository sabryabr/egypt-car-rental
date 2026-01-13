import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import string
import re
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #1e2530; color: white; }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464b5d;
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    label[data-testid="stMetricLabel"] { color: #b0b3b8 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    
    /* Tables */
    .stDataFrame { direction: ltr; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA ENGINE (FIXED & BULLETPROOF) ---
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
                
                # 1. Sanitize Headers
                clean_headers = []
                seen = {}
                for i, h in enumerate(headers):
                    h_str = str(h).strip()
                    if not h_str: h_str = f"Col_{i}" # Name empty columns
                    if h_str in seen:
                        seen[h_str] += 1
                        clean_headers.append(f"{h_str}_{seen[h_str]}")
                    else:
                        seen[h_str] = 0
                        clean_headers.append(h_str)
                
                # 2. Strict Row Enforcement (The Fix)
                target_len = len(clean_headers)
                clean_data = []
                for row in data:
                    # Truncate if too long (fix "30 columns passed" error)
                    row_fixed = row[:target_len] 
                    # Pad if too short
                    if len(row_fixed) < target_len:
                        row_fixed += [None] * (target_len - len(row_fixed))
                    clean_data.append(row_fixed)
                
                return pd.DataFrame(clean_data, columns=clean_headers)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"⚠️ Data Load Error in '{range_name}': {str(e)}")
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
        # Fetching specific ranges to minimize junk data
        dfs['cars'] = fetch_sheet(IDS['cars'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['orders'] = fetch_sheet(IDS['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 1)
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
    """Normalize ID for matching"""
    if pd.isna(x): return "unknown"
    # Remove spaces, specific arabic chars if needed, lower case
    val = str(x).strip().replace(" ", "").lower()
    return val

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
        
        # --- A. PROCESS CARS (Build Map) ---
        car_map = {} # Code -> Label
        active_fleet_count = 0
        
        # Identify columns
        col_code = get_col_by_letter(df_cars, 'A')
        col_type = get_col_by_letter(df_cars, 'B')
        col_model = get_col_by_letter(df_cars, 'E')
        col_year = get_col_by_letter(df_cars, 'H')
        col_plate = get_col_by_letter(df_cars, 'AC') # Start of plate
        col_status = get_col_by_letter(df_cars, 'AZ')

        if col_code and col_status:
            # Filter Valid Cars
            active_cars = df_cars[df_cars[col_status].astype(str).str.contains('Valid|ساري', case=False, na=False)]
            active_fleet_count = len(active_cars)

            for _, row in df_cars.iterrows(): # Map ALL cars, not just active, for history
                try:
                    c_id = clean_id_tag(row[col_code])
                    c_label = f"{row[col_type]} {row[col_model]} ({row[col_year]})"
                    # Try to get plate parts if they exist
                    plate = ""
                    # Combine plate parts roughly (AC to W)
                    for p_col in ['AC','AB','AA','Z','Y','X','W']:
                        c_p = get_col_by_letter(df_cars, p_col)
                        if c_p and pd.notnull(row[c_p]):
                            plate += str(row[c_p]) + " "
                    
                    car_map[c_id] = f"{c_label} | {plate.strip()}"
                except: continue

        # --- B. PROCESS ORDERS ---
        today = datetime.now()
        active_rentals = 0
        returning_today = 0
        departing_today = 0
        timeline_data = []
        
        col_start = get_col_by_letter(df_orders, 'L')
        col_end = get_col_by_letter(df_orders, 'V')
        col_car_ord = get_col_by_letter(df_orders, 'C')
        col_client = get_col_by_letter(df_orders, 'B')
        
        if col_start and col_end and col_car_ord:
            for _, row in df_orders.iterrows():
                try:
                    s_date = pd.to_datetime(row[col_start], errors='coerce')
                    e_date = pd.to_datetime(row[col_end], errors='coerce')
                    
                    if pd.isna(s_date) or pd.isna(e_date): continue
                    
                    # Clean ID to match map
                    raw_car_id = row[col_car_ord]
                    car_id_clean = clean_id_tag(raw_car_id)
                    car_name = car_map.get(car_id_clean, f"Unknown ID ({raw_car_id})")

                    # Metrics
                    if s_date <= today <= e_date: active_rentals += 1
                    if e_date.date() == today.date(): returning_today += 1
                    if s_date.date() == today.date(): departing_today += 1
                    
                    timeline_data.append({
                        'Car': car_name,
                        'Start': s_date,
                        'End': e_date,
                        'Client': str(row[col_client]) if col_client else "N/A",
                        'Status': 'Active' if s_date <= today <= e_date else 'Scheduled'
                    })
                except: continue

        utilization = int((active_rentals / active_fleet_count * 100)) if active_fleet_count > 0 else 0

        # --- C. UI ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚗 Active Rentals", active_rentals, "Live")
        c2.metric("🔄 Returning Today", returning_today, delta_color="inverse")
        c3.metric("🛫 Departing Today", departing_today)
        c4.metric("📊 Utilization", f"{utilization}%", f"{active_fleet_count} Active Cars")
        
        st.divider()
        st.subheader("📅 Live Fleet Schedule")
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            # Filter for logical dates (e.g. 2023 onwards) to avoid clutter
            df_timeline = df_timeline[df_timeline['Start'] > '2023-01-01']

            fig = px.timeline(
                df_timeline, 
                x_start="Start", x_end="End", y="Car", color="Status",
                color_discrete_map={"Active": "#00C853", "Scheduled": "#2962FF"},
                hover_data=["Client"]
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                height=600, 
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"),
                xaxis=dict(showgrid=True, gridcolor="#333"),
            )
            fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid booking dates found. Check 'L' and 'V' columns in Orders sheet.")

# --- 5. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 Vehicle 360° Profile")
    
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    # --- A. SIDEBAR CONTROLS ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Vehicle Filter")
    
    # Build Car List
    car_options = {}
    col_code = get_col_by_letter(df_cars, 'A')
    col_name_parts = ['B', 'E', 'H', 'I'] # Type, Model, Year, Color
    
    if col_code:
        for _, row in df_cars.iterrows():
            try:
                c_id = clean_id_tag(row[col_code])
                c_label = " ".join([str(row[get_col_by_letter(df_cars, c)]) for c in col_name_parts if pd.notnull(row[get_col_by_letter(df_cars, c)])])
                car_options[c_label + f" ({c_id})"] = c_id
            except: continue

    selected_car_label = st.sidebar.selectbox("Select Vehicle", list(car_options.keys()))
    selected_car_id = car_options[selected_car_label]

    # Date Filter for P&L
    sel_year = st.sidebar.selectbox("Analysis Year", [2024, 2025, 2026], index=2)
    sel_month = st.sidebar.selectbox("Analysis Month", range(1, 13), index=0)

    # --- B. GET CAR DETAILS ---
    car_row = df_cars[df_cars[col_code].astype(str).str.strip().str.replace(" ", "").str.lower() == selected_car_id].iloc[0]
    
    # Basic Info
    plate = ""
    for p in ['AC','AB','AA','Z','Y','X','W']:
        val = car_row[get_col_by_letter(df_cars, p)]
        if pd.notnull(val): plate += str(val) + " "
    
    owner_name = f"{car_row[get_col_by_letter(df_cars, 'BP')]} {car_row[get_col_by_letter(df_cars, 'BQ')]}"
    contract_end = car_row[get_col_by_letter(df_cars, 'BC')]
    
    # Financial Config
    base_fee = clean_currency(car_row[get_col_by_letter(df_cars, 'CJ')])
    deduct_pct = clean_currency(car_row[get_col_by_letter(df_cars, 'CL')])
    
    # --- C. CALCULATE FINANCIALS (Month Specific) ---
    # 1. Revenue (Orders)
    month_revenue = 0.0
    trip_count = 0
    col_ord_start = get_col_by_letter(df_orders, 'L')
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_car = get_col_by_letter(df_orders, 'C')

    if col_ord_start:
        # Filter orders for this car AND this month
        for _, row in df_orders.iterrows():
            if clean_id_tag(row[col_ord_car]) == selected_car_id:
                d = pd.to_datetime(row[col_ord_start], errors='coerce')
                if pd.notnull(d) and d.year == sel_year and d.month == sel_month:
                    month_revenue += clean_currency(row[col_ord_cost])
                    trip_count += 1

    # 2. Direct Expenses (Maintenance)
    month_expenses = 0.0
    col_exp_car = get_col_by_letter(df_car_exp, 'S')
    col_exp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_exp_m = get_col_by_letter(df_car_exp, 'X') # Month
    col_exp_y = get_col_by_letter(df_car_exp, 'Y') # Year

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            if clean_id_tag(row[col_exp_car]) == selected_car_id:
                try:
                    if int(clean_currency(row[col_exp_y])) == sel_year and int(clean_currency(row[col_exp_m])) == sel_month:
                        month_expenses += clean_currency(row[col_exp_amt])
                except: continue

    # 3. Estimated Owner Cost (Pro-rated for 1 month)
    # Logic: Base Fee is usually monthly. If Deduction exists, apply it.
    est_owner_cost = base_fee * (1 - (deduct_pct/100))
    
    net_profit = month_revenue - month_expenses - est_owner_cost

    # --- D. VISUAL LAYOUT ---
    
    # Header Card
    with st.container():
        c1, c2, c3 = st.columns([1, 3, 2])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=100)
        with c2:
            st.subheader(f"{selected_car_label}")
            st.caption(f"Plate: {plate} | Owner: {owner_name}")
            st.caption(f"Contract Ends: {contract_end}")
        with c3:
            st.metric("Net Profit (Est.)", f"{net_profit:,.0f} EGP", 
                     delta=f"Rev: {month_revenue:,.0f} | Exp: {month_expenses:,.0f}")

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💰 P&L Analysis", "🛠️ Maintenance Log", "📜 Trip History"])

    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenue (Orders)", f"{month_revenue:,.0f}")
        k2.metric("Maint. Expenses", f"{month_expenses:,.0f}", delta_color="inverse")
        k3.metric("Owner Base Fee", f"{base_fee:,.0f}", help="Before deductions")
        k4.metric("Trips", trip_count)
        
        # Chart: Income vs Expense
        fig = go.Figure(data=[
            go.Bar(name='Revenue', x=['Financials'], y=[month_revenue], marker_color='#2ecc71'),
            go.Bar(name='Direct Exp', x=['Financials'], y=[month_expenses], marker_color='#e74c3c'),
            go.Bar(name='Owner Fee', x=['Financials'], y=[est_owner_cost], marker_color='#f1c40f')
        ])
        fig.update_layout(barmode='group', title=f"Financial Performance: {sel_month}/{sel_year}", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Maintenance & Direct Costs")
        # Filter Expenses for Table
        exp_list = []
        if col_exp_car:
            col_exp_desc = get_col_by_letter(df_car_exp, 'I')
            col_exp_day = get_col_by_letter(df_car_exp, 'W')
            
            for _, row in df_car_exp.iterrows():
                if clean_id_tag(row[col_exp_car]) == selected_car_id:
                    exp_list.append({
                        "Date": f"{row[col_exp_y]}-{row[col_exp_m]}-{row[col_exp_day]}",
                        "Item": row[col_exp_desc],
                        "Cost": clean_currency(row[col_exp_amt])
                    })
        
        if exp_list:
            st.dataframe(pd.DataFrame(exp_list), use_container_width=True)
        else:
            st.info("No expenses recorded for this car.")

    with tab3:
        st.subheader("Trip History")
        # Filter Orders for Table
        trip_list = []
        if col_ord_car:
            col_ord_client = get_col_by_letter(df_orders, 'B')
            col_ord_end = get_col_by_letter(df_orders, 'V')
            
            for _, row in df_orders.iterrows():
                if clean_id_tag(row[col_ord_car]) == selected_car_id:
                    trip_list.append({
                        "Start": row[col_ord_start],
                        "End": row[col_ord_end],
                        "Client": row[col_ord_client],
                        "Revenue": clean_currency(row[col_ord_cost])
                    })
        
        if trip_list:
            st.dataframe(pd.DataFrame(trip_list), use_container_width=True)
        else:
            st.info("No trips recorded.")

# --- 6. PAGE ROUTER ---
if page == "🏠 Operations": show_operations(dfs)
elif page == "🚗 Vehicle 360": show_vehicle_360(dfs)
elif page == "👥 CRM": st.title("👥 CRM (Coming Next)")
elif page == "💰 Financial HQ": st.title("💰 Financial HQ (Coming Next)")
elif page == "⚠️ Risk Radar": st.title("⚠️ Risk Radar (Coming Next)")
