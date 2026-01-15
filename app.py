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
from dateutil.relativedelta import relativedelta
import calendar

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘")

# --- 2. CUSTOM CSS (ULTRA COMPACT) ---
st.markdown("""
<style>
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #0e1117; color: white; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; }
    [data-testid="stSidebar"] { background-color: #1e2530; color: white; }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #262730; border: 1px solid #464b5d; border-radius: 6px; padding: 8px; 
        color: white; height: 85px; overflow: hidden;
    }
    label[data-testid="stMetricLabel"] { font-size: 0.8rem !important; margin-bottom: 0 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    
    /* Compact Elements */
    .stDataFrame { direction: ltr; font-size: 0.85rem; }
    div[data-testid="stExpander"] { border: 1px solid #464b5d; border-radius: 4px; padding: 0px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab"] { height: 35px; padding: 0 10px; font-size: 0.9rem; }
    h1 { font-size: 1.5rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.1rem !important; margin-top: 0.5rem !important; }
    hr { margin: 0.5rem 0 !important; }
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

def get_date_filter_range(period_type, year, specifier):
    if period_type == "Year":
        return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)
    elif period_type == "Quarter":
        q_map = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
        s_m, e_m = q_map[specifier]
        _, last_day = calendar.monthrange(year, e_m)
        return datetime(year, s_m, 1), datetime(year, e_m, last_day, 23, 59, 59)
    else: 
        _, last_day = calendar.monthrange(year, specifier)
        return datetime(year, specifier, 1), datetime(year, specifier, last_day, 23, 59, 59)

# --- 5. MODULE 1: OPERATIONS ---
def show_operations(dfs):
    st.title("🏠 Operations")
    if not dfs: return

    df_orders = dfs['orders']
    df_cars = dfs['cars']

    with st.expander("🔎 Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        period_type = c1.selectbox("Period", ["Month", "Quarter", "Year"])
        sel_year = c2.selectbox("Year", [2024, 2025, 2026, 2027], index=2)
        
        if period_type == "Month":
            sel_spec = c3.selectbox("Month", range(1, 13), index=datetime.now().month-1)
        elif period_type == "Quarter":
            sel_spec = c3.selectbox("Quarter", [1, 2, 3, 4], index=0)
        else: sel_spec = 0 
            
        fleet_status = c4.selectbox("Status", ["Active Only", "All Cars", "Inactive Only"], index=0)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    car_map = {} 
    active_fleet_count = 0
    col_code = get_col_by_letter(df_cars, 'A')
    col_status = get_col_by_letter(df_cars, 'AZ')
    plate_cols = ['AC','AB','AA','Z','Y','X','W']

    if col_code and col_status:
        valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
        
        if fleet_status == "Active Only":
            cars_subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        elif fleet_status == "Inactive Only":
            cars_subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        else:
            cars_subset = valid_rows

        active_fleet_count = len(cars_subset)
        
        for _, row in cars_subset.iterrows(): 
            try:
                c_id = clean_id_tag(row[col_code])
                c_name = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                car_map[c_id] = f"{c_name} | {plate.strip()}"
            except: continue

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
                if not (s_date <= end_range and e_date >= start_range): continue

                car_id_clean = clean_id_tag(row[col_car_ord])
                if car_id_clean not in car_map: continue
                
                status = 'Completed'
                if s_date <= today <= e_date: 
                    status = 'Active'
                    active_rentals += 1
                elif s_date > today: 
                    status = 'Future'
                    future_orders += 1
                
                if e_date.date() == today.date(): returning_today += 1
                
                timeline_data.append({
                    'Car': car_map[car_id_clean], 'Start': s_date, 'End': e_date,
                    'Client': str(row[col_client]) if col_client else "N/A", 'Status': status
                })
            except: continue

    utilization = (active_rentals / active_fleet_count * 100) if active_fleet_count > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Rentals", active_rentals)
    c2.metric("Future", future_orders)
    c3.metric("Returns", returning_today)
    c4.metric("Utilization", f"{utilization:.1f}%")
    
    st.markdown(f"**Schedule ({period_type})**")
    
    all_car_names = sorted(list(car_map.values()))
    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
    else:
        df_timeline = pd.DataFrame(columns=['Car', 'Start', 'End', 'Status', 'Client'])

    for car_name in all_car_names:
        if car_name not in df_timeline['Car'].values:
            new_row = pd.DataFrame([{'Car': car_name, 'Start': pd.NaT, 'End': pd.NaT, 'Status': 'Active', 'Client': ''}])
            df_timeline = pd.concat([df_timeline, new_row], ignore_index=True)

    if not df_timeline.empty:
        color_map = {"Active": "#00C853", "Future": "#9b59b6", "Completed": "#95a5a6"}
        fig = px.timeline(df_timeline, x_start="Start", x_end="End", y="Car", color="Status", color_discrete_map=color_map, hover_data=["Client"])
        fig.update_yaxes(autorange="reversed", categoryorder='array', categoryarray=all_car_names, type='category')
        
        fig.update_layout(
            height=max(300, len(all_car_names) * 30), 
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", 
            font=dict(color="white", size=10), 
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="#333", range=[start_range, end_range])
        )
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("No active fleet.")

# --- 6. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 Vehicle 360")
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    with st.expander("🔎 Vehicle Controls", expanded=True):
        col_filters_1, col_filters_2 = st.columns([1, 2])
        
        with col_filters_1:
            fleet_cat = st.radio("Category", ["Active", "History", "All"], horizontal=True)
            
        with col_filters_2:
            car_options = {}
            col_code = get_col_by_letter(df_cars, 'A')
            col_status = get_col_by_letter(df_cars, 'AZ')
            plate_cols = ['AC','AB','AA','Z','Y','X','W']
            
            if col_code and col_status:
                valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
                
                if fleet_cat == "Active":
                    subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                elif fleet_cat == "History":
                    subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                else:
                    subset = valid_rows 

                for _, row in subset.iterrows():
                    try:
                        c_id = clean_id_tag(row[col_code])
                        c_label = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                        plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                        car_options[f"[{row[col_code]}] {c_label} | {plate.strip()}"] = c_id
                    except: continue

            # SELECT ALL LOGIC
            select_all = st.checkbox("Select All Listed Vehicles")
            default_selection = list(car_options.keys()) if select_all else []
            
            selected_labels = st.multiselect("Vehicles", list(car_options.keys()), default=default_selection)
            selected_ids = [car_options[l] for l in selected_labels]

        st.markdown("---")
        tf1, tf2, tf3, tf4 = st.columns([1, 1, 1, 2])
        with tf1:
            period_type = st.selectbox("View", ["Month", "Quarter", "Year"], key='v360_p')
        with tf2:
            sel_year = st.selectbox("Year", [2024, 2025, 2026], index=2, key='v360_y')
        with tf3:
            if period_type == "Month":
                sel_spec = st.selectbox("Month", range(1, 13), index=datetime.now().month-1, key='v360_m')
            elif period_type == "Quarter":
                sel_spec = st.selectbox("Quarter", [1, 2, 3, 4], index=0, key='v360_q')
            else: sel_spec = 0
        with tf4:
            show_only_active = st.checkbox("Hide empty", value=False)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    if not selected_ids:
        st.info("👈 Select vehicles.")
        return

    trips_data, maint_list, exp_list = [], [], []
    total_revenue, total_maint, total_exp = 0.0, 0.0, 0.0
    
    col_ord_start = get_col_by_letter(df_orders, 'L')
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_car = get_col_by_letter(df_orders, 'C')
    col_ord_client = get_col_by_letter(df_orders, 'B')
    col_ord_id = get_col_by_letter(df_orders, 'A')

    if col_ord_start:
        for _, row in df_orders.iterrows():
            cid = clean_id_tag(row[col_ord_car])
            if cid in selected_ids:
                d = pd.to_datetime(row[col_ord_start], errors='coerce')
                if pd.notnull(d) and start_range <= d <= end_range:
                    rev = clean_currency(row[col_ord_cost])
                    total_revenue += rev
                    trips_data.append({
                        "Car": [k for k, v in car_options.items() if v == cid][0],
                        "Order #": row[col_ord_id], "Start": d, "Client": row[col_ord_client], "Revenue": rev
                    })

    col_exp_car = get_col_by_letter(df_car_exp, 'S')
    col_exp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_exp_day = get_col_by_letter(df_car_exp, 'W')
    col_exp_m = get_col_by_letter(df_car_exp, 'X')
    col_exp_y = get_col_by_letter(df_car_exp, 'Y')
    col_item_maint = get_col_by_letter(df_car_exp, 'I')
    col_item_exp = get_col_by_letter(df_car_exp, 'L')
    col_order_ref = get_col_by_letter(df_car_exp, 'T')

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            cid = clean_id_tag(row[col_exp_car])
            if cid in selected_ids:
                try:
                    y, m = int(clean_currency(row[col_exp_y])), int(clean_currency(row[col_exp_m]))
                    d_val = int(clean_currency(row[col_exp_day]))
                    
                    valid_date = False
                    if period_type == "Year" and y == sel_year: valid_date = True
                    elif period_type == "Month" and y == sel_year and m == sel_spec: valid_date = True
                    elif period_type == "Quarter":
                        q_map = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
                        if y == sel_year and m in q_map[sel_spec]: valid_date = True
                    
                    if valid_date:
                        amt = clean_currency(row[col_exp_amt])
                        is_maint = pd.notnull(row[col_item_maint]) and str(row[col_item_maint]).strip() != ""
                        is_gen_exp = pd.notnull(row[col_item_exp]) and str(row[col_item_exp]).strip() != ""
                        
                        entry = {
                            "Car": [k for k, v in car_options.items() if v == cid][0],
                            "Date": f"{y}-{m}-{d_val}",
                            "Item": str(row[col_item_maint]) if is_maint else str(row[col_item_exp]),
                            "Order Ref": row[col_order_ref], "Cost": amt
                        }
                        if is_maint: 
                            maint_list.append(entry)
                            total_maint += amt
                        elif is_gen_exp: 
                            exp_list.append(entry)
                            total_exp += amt
                except: continue

    if show_only_active and not trips_data:
        st.warning("No data found.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue", f"{total_revenue:,.0f}")
    k2.metric("Maint.", f"{total_maint:,.0f}", delta_color="inverse")
    k3.metric("Exp.", f"{total_exp:,.0f}", delta_color="inverse")
    k4.metric("Yield", f"{total_revenue - total_maint - total_exp:,.0f}")
    
    t1, t2, t3 = st.tabs(["Trips", "Maint.", "Exp."])
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
        else: st.info("No records.")
    with t3:
        if exp_list:
            df_e = pd.DataFrame(exp_list)
            df_e['Cost'] = df_e['Cost'].apply(format_egp)
            st.dataframe(df_e, use_container_width=True)
        else: st.info("No expenses.")

# --- 7. MODULE 3: FINANCIAL HQ ---
def show_financial_hq(dfs):
    st.title("💰 Financial HQ")
    if not dfs: return

    df_cars = dfs['cars']
    df_exp = dfs['expenses']
    df_car_exp = dfs['car_expenses']
    
    # 1. Filters (Expanded logic for Period)
    with st.expander("🗓️ Settings", expanded=True):
        f1, f2, f3 = st.columns(3)
        period_type = f1.selectbox("View", ["Month", "Quarter", "Year"], key='fin_p')
        sel_year = f2.selectbox("Fiscal Year", [2024, 2025, 2026], index=2, key='fin_y')
        
        if period_type == "Month":
            sel_spec = f3.selectbox("Month", range(1, 13), index=0, key='fin_m')
        elif period_type == "Quarter":
            sel_spec = f3.selectbox("Quarter", [1, 2, 3, 4], index=0, key='fin_q')
        else: sel_spec = 0

    start_date, end_date = get_date_filter_range(period_type, sel_year, sel_spec)

    # 2. Advanced Ledger Calculation
    owner_ledger = []
    
    # Columns
    col_code = get_col_by_letter(df_cars, 'A')
    col_status = get_col_by_letter(df_cars, 'AZ')
    col_base = get_col_by_letter(df_cars, 'CJ')
    col_deduct = get_col_by_letter(df_cars, 'CL')
    col_start_contract = get_col_by_letter(df_cars, 'BB') # Assuming Contract Start Date
    col_broker = get_col_by_letter(df_cars, 'CM') # Placeholder for Broker Name (Agent)
    
    # Prepare historical Expenses (Payments)
    # Filter expenses that are "Owner Payouts". Assuming category or keywords.
    # We will search for Car ID in expenses.
    
    payments_map = {} # CarID -> Total Paid History
    current_period_payments = {} # CarID -> Payments in selected period
    
    # Analyze Car Expenses for Payouts
    col_cexp_car = get_col_by_letter(df_car_exp, 'S')
    col_cexp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_cexp_type = get_col_by_letter(df_car_exp, 'F')
    col_cexp_y = get_col_by_letter(df_car_exp, 'Y')
    col_cexp_m = get_col_by_letter(df_car_exp, 'X')
    col_cexp_d = get_col_by_letter(df_car_exp, 'W')

    if col_cexp_car:
        for _, row in df_car_exp.iterrows():
            try:
                cid = clean_id_tag(row[col_cexp_car])
                amt = clean_currency(row[col_cexp_amt])
                exp_type = str(row[col_cexp_type]).lower()
                
                # Check if this is an Owner Payment (adjust keyword as needed)
                is_payout = "owner" in exp_type or "payout" in exp_type or "مالك" in exp_type or "دفع" in exp_type
                
                if is_payout:
                    payments_map[cid] = payments_map.get(cid, 0) + amt
                    
                    # Check if in current period
                    y, m, d = int(clean_currency(row[col_cexp_y])), int(clean_currency(row[col_cexp_m])), int(clean_currency(row[col_cexp_d]))
                    exp_date = datetime(y, m, d)
                    if start_date <= exp_date <= end_date:
                        current_period_payments[cid] = current_period_payments.get(cid, 0) + amt
            except: continue

    # Calculate Accruals (What we owe)
    for _, car in df_cars.iterrows():
        try:
            if col_status and not any(x in str(car[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
            
            cid = clean_id_tag(car[col_code])
            base_fee = clean_currency(car[col_base])
            deduct_pct = clean_currency(car[col_deduct])
            
            # Start Date logic
            start_contract = pd.to_datetime(car[col_start_contract], errors='coerce')
            if pd.isna(start_contract): start_contract = datetime(2023, 1, 1) # Fallback
            
            # Calculate DUE DATE for current period
            # If period is Month 1, and contract started on 15th, due date is 1/15
            due_day = start_contract.day
            
            # Ensure due day is valid for the selected month (e.g. Feb 30 -> Feb 28)
            try:
                period_due_date = datetime(sel_year, sel_spec if period_type=="Month" else 1, due_day)
            except ValueError:
                period_due_date = datetime(sel_year, sel_spec, 28) # Simple fallback
            
            # Commission Logic (Check if this owner is a broker for others)
            # This is complex without a clear link. I will add a placeholder column.
            commission = 0.0 
            
            # Gross Monthly (Net after operations fee)
            monthly_net = base_fee * (1 - (deduct_pct/100))
            
            # Current Month Dues
            if period_type == "Month":
                dues_this_period = monthly_net
                due_date_str = period_due_date.strftime("%Y-%m-%d")
            else:
                dues_this_period = monthly_net * 3 # Approx for quarter
                due_date_str = "Various"

            # Historical Balance Calculation
            # Calculate months passed since start until NOW
            months_active = (datetime.now().year - start_contract.year) * 12 + (datetime.now().month - start_contract.month)
            total_accrued_lifetime = months_active * monthly_net
            total_paid_lifetime = payments_map.get(cid, 0)
            
            # Current Balance (Positive = We owe them, Negative = We overpaid)
            current_balance = total_accrued_lifetime - total_paid_lifetime

            owner_ledger.append({
                "Car": f"{car[get_col_by_letter(df_cars, 'B')]} {car[get_col_by_letter(df_cars, 'E')]}",
                "Due Date": due_date_str,
                "Monthly Fee": base_fee,
                "Net Due (Period)": dues_this_period,
                "Commission": commission,
                "Paid (Period)": current_period_payments.get(cid, 0),
                "Total Balance (Lifetime)": current_balance
            })

        except: continue

    st.subheader(f"Owner Statement: {period_type} {sel_spec if period_type!='Year' else ''}/{sel_year}")
    
    if owner_ledger:
        df_l = pd.DataFrame(owner_ledger)
        # Format
        for c in ["Monthly Fee", "Net Due (Period)", "Commission", "Paid (Period)", "Total Balance (Lifetime)"]:
            df_l[c] = df_l[c].apply(format_egp)
        
        # Highlight Balance
        def color_balance(val):
            val_num = float(val.replace(" EGP", "").replace(",", ""))
            color = 'red' if val_num > 0 else 'green' # Red means we owe money (Debt)
            return f'color: {color}'

        st.dataframe(df_l.style.map(color_balance, subset=['Total Balance (Lifetime)']), use_container_width=True, height=500)
    else:
        st.info("No active contracts.")

# --- 8. PAGE ROUTER ---
st.sidebar.title("🚘 Rental OS")
page = st.sidebar.radio("Nav", ["Operations", "Vehicle 360", "Financial HQ"])
st.sidebar.markdown("---")

dfs = load_data_v3()

if page == "Operations": show_operations(dfs)
elif page == "Vehicle 360": show_vehicle_360(dfs)
elif page == "Financial HQ": show_financial_hq(dfs)
