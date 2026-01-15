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

# --- 3. DATA ENGINE ---
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
                for i, h in enumerate(headers):
                    h_str = str(h).strip()
                    if not h_str: h_str = f"Col_{i}"
                    if h_str in seen:
                        seen[h_str] += 1
                        clean_headers.append(f"{h_str}_{seen[h_str]}")
                    else:
                        seen[h_str] = 0
                        clean_headers.append(h_str)
                
                target_len = len(clean_headers)
                clean_data = []
                for row in data:
                    row_fixed = row[:target_len] 
                    if len(row_fixed) < target_len:
                        row_fixed += [None] * (target_len - len(row_fixed))
                    clean_data.append(row_fixed)
                
                return pd.DataFrame(clean_data, columns=clean_headers)
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"⚠️ Load Error ({range_name}): {str(e)}")
            return pd.DataFrame()

    IDS = {
        'cars': "1tQVkPj7tCnrKsHEIs04a1WzzC04jpOWuLsXgXOkVMkk", 
        'orders': "1T6j2xnRBTY31crQcJHioKurs4Rvaj-VlEQkm6joGxGM",
        'clients': "13YZOGdRCEy7IMZHiTmjLFyO417P8dD0m5Sh9xwKI8js",
        'expenses': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_expenses': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'collections': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg"
    }

    with st.spinner("🔄 Syncing HQ Data..."):
        dfs = {}
        dfs['cars'] = fetch_sheet(IDS['cars'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['orders'] = fetch_sheet(IDS['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 1)
        dfs['clients'] = fetch_sheet(IDS['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['expenses'] = fetch_sheet(IDS['expenses'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['car_expenses'] = fetch_sheet(IDS['car_expenses'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        dfs['collections'] = fetch_sheet(IDS['collections'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
        return dfs

# --- 4. HELPER FUNCTIONS ---
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
    if pd.isna(x): return "unknown"
    return str(x).strip().replace(" ", "").lower()

def clean_currency(x):
    if pd.isna(x): return 0.0
    s = str(x).replace(',', '').replace('%', '').strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

def format_egp(x):
    return f"{x:,.0f} EGP"

# --- 5. FILTER HELPER ---
def get_date_filter_range(period_type, year, specifier):
    """Returns start and end datetime for Month/Quarter/Year"""
    if period_type == "Year":
        return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)
    elif period_type == "Quarter":
        q_map = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
        s_m, e_m = q_map[specifier]
        # End day logic
        _, last_day =  (31 if e_m in [1,3,5,7,8,10,12] else 30, 31) # Simple approx, better to use calendar lib if strict
        # Using calendar for robustness
        import calendar
        _, last_day = calendar.monthrange(year, e_m)
        return datetime(year, s_m, 1), datetime(year, e_m, last_day, 23, 59, 59)
    else: # Month
        import calendar
        _, last_day = calendar.monthrange(year, specifier)
        return datetime(year, specifier, 1), datetime(year, specifier, last_day, 23, 59, 59)

# --- 6. MODULE 1: OPERATIONS (DEFAULT: ACTIVE ONLY) ---
def show_operations(dfs):
    st.title("🏠 Operations Command Center")
    if not dfs: return

    df_orders = dfs['orders']
    df_cars = dfs['cars']

    # --- A. FILTERS ---
    with st.expander("🔎 Filters & View Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        period_type = c1.selectbox("Period Type", ["Month", "Quarter", "Year"])
        sel_year = c2.selectbox("Year", [2024, 2025, 2026, 2027], index=2)
        
        if period_type == "Month":
            sel_spec = c3.selectbox("Month", range(1, 13), index=datetime.now().month-1)
        elif period_type == "Quarter":
            sel_spec = c3.selectbox("Quarter", [1, 2, 3, 4], index=0)
        else:
            sel_spec = 0 
            
        # UPDATE: "Active Only" is now the default (Index 0)
        fleet_status = c4.selectbox("Fleet Filter", ["Active Only", "All Cars", "Inactive Only"], index=0)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    # --- B. PROCESS CARS ---
    car_map = {} 
    active_fleet_count = 0
    
    col_code = get_col_by_letter(df_cars, 'A')
    col_type = get_col_by_letter(df_cars, 'B')
    col_model = get_col_by_letter(df_cars, 'E')
    col_year = get_col_by_letter(df_cars, 'H')
    col_status = get_col_by_letter(df_cars, 'AZ')
    plate_cols = ['AC','AB','AA','Z','Y','X','W']

    if col_code and col_status:
        # 1. CLEAN: Remove empty rows (Fixes the "516 Cars" bug)
        valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
        
        # 2. FILTER: Apply the Active/Inactive logic
        if fleet_status == "Active Only":
            # Only keeps cars marked 'Valid' or 'ساري'
            cars_subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|ساري', case=False, na=False)]
        elif fleet_status == "Inactive Only":
            cars_subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|ساري', case=False, na=False)]
        else:
            cars_subset = valid_rows

        active_fleet_count = len(cars_subset)
        
        # 3. MAP: Build the list of visible cars
        for _, row in cars_subset.iterrows(): 
            try:
                c_id = clean_id_tag(row[col_code])
                c_name = f"{row[col_type]} {row[col_model]} ({row[col_year]})"
                
                plate = ""
                for p in plate_cols:
                    val = row[get_col_by_letter(df_cars, p)]
                    if pd.notnull(val): plate += str(val) + " "
                
                car_map[c_id] = f"[{row[col_code]}] {c_name} | {plate.strip()}"
            except: continue

    # --- C. PROCESS ORDERS ---
    today = datetime.now()
    active_rentals = 0
    returning_today = 0
    future_orders = 0
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
                
                # Date Range Check
                if not (s_date <= end_range and e_date >= start_range):
                    continue

                raw_car_id = row[col_car_ord]
                car_id_clean = clean_id_tag(raw_car_id)
                
                # CRITICAL: If car was filtered out (inactive), skip its order
                if car_id_clean not in car_map: continue
                
                car_name = car_map[car_id_clean]

                # Status Logic
                status = 'Completed'
                if s_date <= today <= e_date: 
                    status = 'Active'
                    active_rentals += 1
                elif s_date > today: 
                    status = 'Future'
                    future_orders += 1
                
                if e_date.date() == today.date(): returning_today += 1
                
                timeline_data.append({
                    'Car': car_name, 'Start': s_date, 'End': e_date,
                    'Client': str(row[col_client]) if col_client else "N/A",
                    'Status': status
                })
            except: continue

    # Utilization Math
    utilization = (active_rentals / active_fleet_count * 100) if active_fleet_count > 0 else 0.0

    # --- D. UI VISUALS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚗 Active Rentals", active_rentals, "Live")
    c2.metric("📅 Future Bookings", future_orders, "Paid & Pending")
    c3.metric("🔄 Returning Today", returning_today, delta_color="inverse")
    c4.metric("📊 Utilization", f"{utilization:.1f}%", f"{active_fleet_count} Active Cars")
    
    st.divider()
    st.subheader(f"📅 Fleet Schedule ({period_type} View)")
    
    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        
        color_map = {
            "Active": "#00C853",   # Green
            "Future": "#9b59b6",   # Purple
            "Completed": "#95a5a6" # Grey
        }

        fig = px.timeline(
            df_timeline, 
            x_start="Start", x_end="End", y="Car", color="Status",
            color_discrete_map=color_map,
            hover_data=["Client"]
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=max(400, len(car_map)*30), # Auto-height based on number of cars
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(showgrid=True, gridcolor="#333", range=[start_range, end_range]),
        )
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
        st.plotly_chart(fig, use_container_width=True)
    else: 
        if active_fleet_count == 0:
            st.warning("No active cars found. Check your 'Cars' sheet column AZ for 'Valid' or 'ساري'.")
        else:
            st.info("No bookings found for the active fleet in this period.")

# --- 7. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 Vehicle 360° Profile")
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    # --- FILTERS ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")
    
    # 1. Car Builder
    car_options = {}
    col_code = get_col_by_letter(df_cars, 'A')
    
    # Plate columns
    plate_cols = ['AC','AB','AA','Z','Y','X','W']
    
    if col_code:
        for _, row in df_cars.iterrows():
            try:
                c_id = clean_id_tag(row[col_code])
                c_label = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                
                plate = ""
                for p in plate_cols:
                    val = row[get_col_by_letter(df_cars, p)]
                    if pd.notnull(val): plate += str(val) + " "
                
                car_options[f"[{row[col_code]}] {c_label} | {plate.strip()}"] = c_id
            except: continue

    # Multi-Select
    selected_labels = st.sidebar.multiselect("Select Vehicles", list(car_options.keys()))
    selected_ids = [car_options[l] for l in selected_labels]
    
    # Time Filters
    period_type = st.sidebar.selectbox("Period Type", ["Month", "Quarter", "Year"], key='v360_p')
    sel_year = st.sidebar.selectbox("Year", [2024, 2025, 2026], index=2, key='v360_y')
    if period_type == "Month":
        sel_spec = st.sidebar.selectbox("Month", range(1, 13), index=datetime.now().month-1, key='v360_m')
    elif period_type == "Quarter":
        sel_spec = st.sidebar.selectbox("Quarter", [1, 2, 3, 4], index=0, key='v360_q')
    else:
        sel_spec = 0

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    # Filter Toggle
    show_only_active = st.sidebar.checkbox("Show only cars with trips in period", value=False)

    if not selected_ids:
        st.info("Please select at least one vehicle from the sidebar.")
        return

    # --- PROCESSING ---
    
    # 1. Get Revenue & Trips
    trips_data = []
    total_revenue = 0.0
    
    col_ord_start = get_col_by_letter(df_orders, 'L')
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_car = get_col_by_letter(df_orders, 'C')
    col_ord_client = get_col_by_letter(df_orders, 'B')
    col_ord_id = get_col_by_letter(df_orders, 'A')

    active_cars_in_period = set()

    if col_ord_start:
        for _, row in df_orders.iterrows():
            cid = clean_id_tag(row[col_ord_car])
            if cid in selected_ids:
                d = pd.to_datetime(row[col_ord_start], errors='coerce')
                if pd.notnull(d) and start_range <= d <= end_range:
                    rev = clean_currency(row[col_ord_cost])
                    total_revenue += rev
                    active_cars_in_period.add(cid)
                    
                    trips_data.append({
                        "Car": [k for k, v in car_options.items() if v == cid][0], # Reverse lookup name
                        "Order #": row[col_ord_id],
                        "Start": d,
                        "Client": row[col_ord_client],
                        "Revenue": rev
                    })

    # 2. Get Expenses (Split)
    maint_list = []
    exp_list = []
    total_maint = 0.0
    total_exp = 0.0
    
    col_exp_car = get_col_by_letter(df_car_exp, 'S')
    col_exp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_exp_day = get_col_by_letter(df_car_exp, 'W')
    col_exp_m = get_col_by_letter(df_car_exp, 'X')
    col_exp_y = get_col_by_letter(df_car_exp, 'Y')
    
    # Columns for Split Logic
    col_item_maint = get_col_by_letter(df_car_exp, 'I') # Maintenance
    col_item_exp = get_col_by_letter(df_car_exp, 'L')   # General Expense (Petty Cash)
    col_order_ref = get_col_by_letter(df_car_exp, 'T')  # Order Num

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            cid = clean_id_tag(row[col_exp_car])
            if cid in selected_ids:
                try:
                    # Date Check
                    y = int(clean_currency(row[col_exp_y]))
                    m = int(clean_currency(row[col_exp_m]))
                    d = int(clean_currency(row[col_exp_day]))
                    # Approx check (constructing full date might fail if day is invalid)
                    # We'll rely on Year/Month for Month filter, Year for Year.
                    # For Quarter/Range, we need full date.
                    
                    # Simple Filter Logic
                    valid_date = False
                    if period_type == "Year":
                        if y == sel_year: valid_date = True
                    elif period_type == "Month":
                        if y == sel_year and m == sel_spec: valid_date = True
                    elif period_type == "Quarter":
                        # Simplification: check if month is in quarter
                        q_map = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
                        if y == sel_year and m in q_map[sel_spec]: valid_date = True

                    if valid_date:
                        amt = clean_currency(row[col_exp_amt])
                        
                        # Split Logic
                        is_maint = pd.notnull(row[col_item_maint]) and str(row[col_item_maint]).strip() != ""
                        is_gen_exp = pd.notnull(row[col_item_exp]) and str(row[col_item_exp]).strip() != ""
                        
                        entry = {
                            "Car": [k for k, v in car_options.items() if v == cid][0],
                            "Date": f"{y}-{m}-{d}",
                            "Item": str(row[col_item_maint]) if is_maint else str(row[col_item_exp]),
                            "Order Ref": row[col_order_ref],
                            "Cost": amt
                        }
                        
                        if is_maint:
                            maint_list.append(entry)
                            total_maint += amt
                        elif is_gen_exp:
                            exp_list.append(entry)
                            total_exp += amt
                            
                except: continue

    # Filter Enforcement
    if show_only_active and not trips_data:
        st.warning("No trips found for selected vehicles in this period.")
        return

    # --- UI ---
    st.subheader(f"📊 Fleet Performance ({len(selected_ids)} Vehicles)")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"{total_revenue:,.0f} EGP")
    k2.metric("Maintenance", f"{total_maint:,.0f} EGP", delta_color="inverse")
    k3.metric("Other Expenses", f"{total_exp:,.0f} EGP", delta_color="inverse")
    k4.metric("Net Yield", f"{total_revenue - total_maint - total_exp:,.0f} EGP")

    st.divider()
    
    t1, t2, t3 = st.tabs(["📜 Trip Details", "🛠️ Maintenance Log", "💸 Expense Log"])
    
    with t1:
        if trips_data:
            df_t = pd.DataFrame(trips_data)
            df_t['Revenue'] = df_t['Revenue'].apply(format_egp)
            st.dataframe(df_t, use_container_width=True)
        else: st.info("No trips.")
        
    with t2:
        if maint_list:
            df_m = pd.DataFrame(maint_list)
            df_m['Cost'] = df_m['Cost'].apply(format_egp)
            st.dataframe(df_m, use_container_width=True)
        else: st.info("No maintenance records.")
        
    with t3:
        if exp_list:
            df_e = pd.DataFrame(exp_list)
            df_e['Cost'] = df_e['Cost'].apply(format_egp)
            st.dataframe(df_e, use_container_width=True)
        else: st.info("No other expenses.")

# --- 8. PAGE ROUTER ---
st.sidebar.title("🚘 Rental OS 3.0")
page = st.sidebar.radio("Navigate", ["🏠 Operations", "🚗 Vehicle 360", "👥 CRM", "💰 Financial HQ", "⚠️ Risk Radar"])
st.sidebar.markdown("---")

dfs = load_data_v3()

if page == "🏠 Operations": show_operations(dfs)
elif page == "🚗 Vehicle 360": show_vehicle_360(dfs)
elif page == "👥 CRM": st.title("👥 CRM (Coming Next)")
elif page == "💰 Financial HQ": 
    # Use existing Financial Logic (Placeholder for now to save space, paste previous module here if needed)
    st.info("Financial Module Loaded (See V19 code)")
elif page == "⚠️ Risk Radar": st.title("⚠️ Risk Radar (Coming Next)")
