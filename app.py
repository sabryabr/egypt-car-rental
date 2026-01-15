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
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘", initial_sidebar_state="auto")

# --- 2. ULTRA-COMPACT CSS ---
st.markdown("""
<style>
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #0e1117; color: white; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 3rem !important; }
    [data-testid="stSidebar"] { background-color: #1e2530; color: white; }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #262730; border: 1px solid #464b5d; border-radius: 6px; padding: 5px 10px; 
        color: white; height: auto; min-height: 70px; overflow: hidden;
    }
    label[data-testid="stMetricLabel"] { font-size: 0.75rem !important; margin-bottom: 0 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    
    /* Tables */
    .stDataFrame { direction: ltr; font-size: 0.8rem; }
    div[data-testid="stExpander"] { border: 1px solid #464b5d; border-radius: 4px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; margin-bottom: 0.5rem; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { height: 35px; padding: 0 10px; font-size: 0.85rem; flex-grow: 1; }
    
    /* Mobile */
    @media (max-width: 640px) {
        div[data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: 100px !important; }
    }
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

def clean_client_code(x):
    if pd.isna(x): return "unknown"
    s = str(x).strip()
    if s.endswith(".0"): s = s[:-2]
    return s

def clean_currency(x):
    if pd.isna(x): return 0.0
    s = str(x).replace(',', '').replace('%', '').strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

def format_egp(x):
    if x >= 1000000: return f"{x/1000000:.1f}M"
    if x >= 1000: return f"{x/1000:.1f}k"
    return f"{x:,.0f}"

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
        c1, c2 = st.columns(2)
        period_type = c1.selectbox("Period", ["Month", "Quarter", "Year"])
        sel_year = c2.selectbox("Year", [2024, 2025, 2026, 2027], index=2)
        c3, c4 = st.columns(2)
        if period_type == "Month":
            sel_spec = c3.selectbox("Month", range(1, 13), index=datetime.now().month-1)
        elif period_type == "Quarter":
            sel_spec = c3.selectbox("Quarter", [1, 2, 3, 4], index=0)
        else: sel_spec = 0 
        fleet_status = c4.selectbox("Status", ["Active Only", "All Cars", "Inactive Only"], index=0)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    car_map = {} 
    active_fleet_count = 0
    sunburst_data = []
    
    col_code = get_col_by_letter(df_cars, 'A')
    col_status = get_col_by_letter(df_cars, 'AZ')
    col_brand = get_col_by_letter(df_cars, 'B')
    col_model = get_col_by_letter(df_cars, 'E')
    plate_cols = ['AC','AB','AA','Z','Y','X','W']

    if col_code and col_status:
        valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
        if fleet_status == "Active Only":
            cars_subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        elif fleet_status == "Inactive Only":
            cars_subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        else: cars_subset = valid_rows

        active_fleet_count = len(cars_subset)
        for _, row in cars_subset.iterrows(): 
            try:
                c_id = clean_id_tag(row[col_code])
                c_name = f"{row[col_brand]} {row[col_model]}"
                plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                car_map[c_id] = f"{c_name} | {plate.strip()}"
                sunburst_data.append({'Brand': str(row[col_brand]).strip(), 'Model': str(row[col_model]).strip(), 'Count': 1})
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

    available_cars = active_fleet_count - active_rentals
    utilization = (active_rentals / active_fleet_count * 100) if active_fleet_count > 0 else 0.0
    
    st.subheader("📊 Fleet Pulse")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚗 Total", active_fleet_count)
    k2.metric("⚡ Live", active_rentals)
    k3.metric("🟢 Free", available_cars)
    k4.metric("📈 Util", f"{utilization:.1f}%")
    
    c1, c2 = st.columns(2)
    with c1:
        if sunburst_data:
            fig = px.sunburst(pd.DataFrame(sunburst_data), path=['Brand', 'Model'], values='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=250, margin=dict(t=0, l=0, r=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(names=['Rented', 'Available'], values=[active_rentals, available_cars], hole=0.5, color_discrete_map={'Rented':'#00C853', 'Available':'#29b6f6'})
        fig.update_layout(height=250, margin=dict(t=0, l=0, r=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(f"**Schedule ({period_type})**")
    
    all_car_names = sorted(list(car_map.values()))
    df_timeline = pd.DataFrame(timeline_data) if timeline_data else pd.DataFrame(columns=['Car', 'Start', 'End', 'Status', 'Client'])

    for car_name in all_car_names:
        if car_name not in df_timeline['Car'].values:
            new_row = pd.DataFrame([{'Car': car_name, 'Start': pd.NaT, 'End': pd.NaT, 'Status': 'Active', 'Client': ''}])
            df_timeline = pd.concat([df_timeline, new_row], ignore_index=True)

    if not df_timeline.empty:
        color_map = {"Active": "#00C853", "Future": "#9b59b6", "Completed": "#95a5a6"}
        fig = px.timeline(df_timeline, x_start="Start", x_end="End", y="Car", color="Status", color_discrete_map=color_map, hover_data=["Client"])
        fig.update_yaxes(autorange="reversed", categoryorder='array', categoryarray=all_car_names, type='category')
        fig.update_layout(height=max(300, len(all_car_names) * 35), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", 
                          font=dict(color="white", size=10), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(showgrid=True, gridcolor="#333", range=[start_range, end_range]))
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("No fleet.")

# --- 6. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 Vehicle 360")
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    with st.expander("🔎 Controls", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1: fleet_cat = st.radio("Category", ["Active", "History", "All"], horizontal=True)
        with col2:
            car_options = {}
            col_code = get_col_by_letter(df_cars, 'A')
            col_status = get_col_by_letter(df_cars, 'AZ')
            plate_cols = ['AC','AB','AA','Z','Y','X','W']
            if col_code and col_status:
                valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
                if fleet_cat == "Active": subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                elif fleet_cat == "History": subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                else: subset = valid_rows 
                for _, row in subset.iterrows():
                    try:
                        c_id = clean_id_tag(row[col_code])
                        c_label = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                        plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                        car_options[f"[{row[col_code]}] {c_label} | {plate.strip()}"] = c_id
                    except: continue
            select_all = st.checkbox("Select All")
            default_sel = list(car_options.keys()) if select_all else []
            selected_labels = st.multiselect("Vehicles", list(car_options.keys()), default=default_sel)
            selected_ids = [car_options[l] for l in selected_labels]

        st.markdown("---")
        tf1, tf2 = st.columns(2)
        period_type = tf1.selectbox("View", ["Month", "Quarter", "Year"], key='v360_p')
        sel_year = tf2.selectbox("Year", [2024, 2025, 2026], index=2, key='v360_y')
        tf3, tf4 = st.columns(2)
        if period_type == "Month": sel_spec = tf3.selectbox("Month", range(1, 13), index=datetime.now().month-1, key='v360_m')
        elif period_type == "Quarter": sel_spec = tf3.selectbox("Quarter", [1, 2, 3, 4], index=0, key='v360_q')
        else: sel_spec = 0
        show_active = tf4.checkbox("Hide empty", value=False)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)
    if not selected_ids: st.info("👈 Select vehicles."); return

    trips_data, maint_list, exp_list = [], [], []
    total_revenue, total_maint, total_exp = 0.0, 0.0, 0.0
    
    col_ord_start = get_col_by_letter(df_orders, 'L')
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_car = get_col_by_letter(df_orders, 'C')
    col_ord_id = get_col_by_letter(df_orders, 'A')

    if col_ord_start:
        for _, row in df_orders.iterrows():
            cid = clean_id_tag(row[col_ord_car])
            if cid in selected_ids:
                d = pd.to_datetime(row[col_ord_start], errors='coerce')
                if pd.notnull(d) and start_range <= d <= end_range:
                    rev = clean_currency(row[col_ord_cost])
                    total_revenue += rev
                    trips_data.append({"Car": [k for k, v in car_options.items() if v == cid][0], "Order": row[col_ord_id], "Date": d, "Rev": rev})

    col_exp_car = get_col_by_letter(df_car_exp, 'S')
    col_exp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_exp_y = get_col_by_letter(df_car_exp, 'Y')
    col_exp_m = get_col_by_letter(df_car_exp, 'X')
    col_item_maint = get_col_by_letter(df_car_exp, 'I')
    col_item_exp = get_col_by_letter(df_car_exp, 'L')

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            cid = clean_id_tag(row[col_exp_car])
            if cid in selected_ids:
                try:
                    y, m = int(clean_currency(row[col_exp_y])), int(clean_currency(row[col_exp_m]))
                    valid = False
                    if period_type=="Year" and y==sel_year: valid=True
                    elif period_type=="Month" and y==sel_year and m==sel_spec: valid=True
                    elif period_type=="Quarter":
                         if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: valid=True
                    if valid:
                        amt = clean_currency(row[col_exp_amt])
                        is_maint = pd.notnull(row[col_item_maint]) and str(row[col_item_maint]).strip() != ""
                        item = str(row[col_item_maint]) if is_maint else str(row[col_item_exp])
                        entry = {"Car": [k for k, v in car_options.items() if v == cid][0], "Date": f"{y}-{m}", "Item": item, "Cost": amt}
                        if is_maint: maint_list.append(entry); total_maint += amt
                        else: exp_list.append(entry); total_exp += amt
                except: continue

    if show_active and not trips_data: st.warning("No data."); return

    k1, k2 = st.columns(2)
    k1.metric("Revenue", format_egp(total_revenue))
    k2.metric("Maint.", format_egp(total_maint), delta_color="inverse")
    k3, k4 = st.columns(2)
    k3.metric("Exp.", format_egp(total_exp), delta_color="inverse")
    k4.metric("Yield", format_egp(total_revenue - total_maint - total_exp))
    
    t1, t2, t3 = st.tabs(["Trips", "Maint.", "Exp."])
    with t1:
        if trips_data: st.dataframe(pd.DataFrame(trips_data), use_container_width=True)
        else: st.info("Empty")
    with t2:
        if maint_list: st.dataframe(pd.DataFrame(maint_list), use_container_width=True)
        else: st.info("Empty")
    with t3:
        if exp_list: st.dataframe(pd.DataFrame(exp_list), use_container_width=True)
        else: st.info("Empty")

# --- 7. MODULE 3: CRM ---
def show_crm(dfs):
    st.title("👥 CRM")
    if not dfs: return
    
    df_orders = dfs['orders']
    df_clients = dfs['clients'] 
    df_cars = dfs['cars']

    # 1. Car Map
    car_display_map = {}
    col_code = get_col_by_letter(df_cars, 'A')
    plate_cols = ['AC','AB','AA','Z','Y','X','W']
    if col_code:
        for _, row in df_cars.iterrows():
            try:
                cid = clean_id_tag(row[col_code])
                cname = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                car_display_map[cid] = f"{cname} | {plate.strip()}"
            except: continue

    # 2. Client Map
    client_id_map = {} 
    client_db = {}
    col_cl_id = get_col_by_letter(df_clients, 'A')
    col_cl_first = get_col_by_letter(df_clients, 'C') 
    col_cl_last = get_col_by_letter(df_clients, 'D')
    
    if col_cl_id:
        for _, row in df_clients.iterrows():
            try:
                cid = clean_client_code(row[col_cl_id])
                fname = str(row[col_cl_first]) if pd.notnull(row[col_cl_first]) else ""
                lname = str(row[col_cl_last]) if pd.notnull(row[col_cl_last]) else ""
                full_name = f"{fname} {lname}".strip()
                if not full_name: continue
                client_id_map[cid] = full_name
                client_db[full_name] = {'Display': f"[{cid}] {full_name}", 'Name': full_name, 'Spend': 0, 'Trips': 0, 'History': []}
            except: continue

    # 3. Process Orders
    col_ord_name = get_col_by_letter(df_orders, 'B')
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_s = get_col_by_letter(df_orders, 'L')
    col_ord_e = get_col_by_letter(df_orders, 'V')
    col_ord_car = get_col_by_letter(df_orders, 'C')
    col_ord_id = get_col_by_letter(df_orders, 'A')

    if col_ord_name:
        for _, row in df_orders.iterrows():
            try:
                raw_val = clean_client_code(row[col_ord_name])
                if not raw_val or raw_val == "nan": continue
                real_name = client_id_map.get(raw_val, raw_val) 
                if real_name not in client_db:
                    client_db[real_name] = {'Display': f"[?] {real_name}", 'Name': real_name, 'Spend': 0, 'Trips': 0, 'History': []}
                
                rec = client_db[real_name]
                amt = clean_currency(row[col_ord_cost])
                s = pd.to_datetime(row[col_ord_s], errors='coerce')
                e = pd.to_datetime(row[col_ord_e], errors='coerce')
                cid = clean_id_tag(row[col_ord_car])
                status = "Completed"
                days = 0
                if pd.notnull(s) and pd.notnull(e):
                    days = (e - s).days
                    if s <= datetime.now() <= e: status = "Active"
                    elif s > datetime.now(): status = "Future"
                
                daily_rate = (amt / days) if days > 0 else 0
                rec['Spend'] += amt
                rec['Trips'] += 1
                rec['History'].append({
                    "Order ID": row[col_ord_id],
                    "Car": car_display_map.get(cid, f"Unknown ({cid})"),
                    "Start": s.strftime("%Y-%m-%d") if pd.notnull(s) else "-",
                    "End": e.strftime("%Y-%m-%d") if pd.notnull(e) else "-",
                    "Days": days,
                    "Cost": format_egp(amt),
                    "Daily Rate": format_egp(daily_rate),
                    "Status": status
                })
            except: continue

    # UI
    search = st.text_input("🔍 Search Client", "")
    data_list = []
    for k, v in client_db.items():
        data_list.append({'Display': v['Display'], 'Spend': v['Spend'], 'Trips': v['Trips'], 'Key': v['Name']})
    
    df_crm = pd.DataFrame(data_list)
    if not df_crm.empty:
        df_crm = df_crm.sort_values('Spend', ascending=False)
        if search: df_crm = df_crm[df_crm['Display'].str.contains(search, case=False, na=False)]

        c1, c2, c3 = st.columns(3)
        c1.metric("Clients", len(client_db))
        c2.metric("Top", df_crm.iloc[0]['Display'].split("] ")[1] if "]" in df_crm.iloc[0]['Display'] else df_crm.iloc[0]['Display'])
        c3.metric("Avg LTV", format_egp(df_crm['Spend'].mean()))

        st.divider()
        col_list, col_detail = st.columns([1, 2])
        
        with col_list:
            df_display = df_crm.copy()
            df_display['Spend'] = df_display['Spend'].apply(format_egp)
            selection = st.dataframe(df_display[['Display', 'Spend', 'Trips']], use_container_width=True, height=500, on_select="rerun", selection_mode="single-row", hide_index=True)
        
        with col_detail:
            sel_idx = selection.selection.rows
            if sel_idx:
                client_key = df_display.iloc[sel_idx[0]]['Key']
                client_data = client_db[client_key]
                st.info(f"**{client_data['Display']}**")
                m1, m2 = st.columns(2)
                m1.metric("Total", format_egp(client_data['Spend']))
                m2.metric("Trips", client_data['Trips'])
                
                if client_data['History']:
                    hist_df = pd.DataFrame(client_data['History'])
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)
                else: st.warning("No history.")
            else: st.info("👈 Select a client.")

# --- 8. MODULE 4: FINANCIAL HQ (ULTIMATE LEDGER FIX) ---
def show_financial_hq(dfs):
    st.title("💰 Financial HQ")
    if not dfs: return

    df_coll = dfs['collections']
    df_exp = dfs['expenses']
    df_car_exp = dfs['car_expenses']
    df_cars = dfs['cars']
    df_orders = dfs['orders']

    with st.expander("🗓️ Settings", expanded=True):
        f1, f2 = st.columns(2)
        period_type = f1.selectbox("View", ["Month", "Quarter", "Year"], key='fin_p')
        sel_year = f2.selectbox("Fiscal Year", [2024, 2025, 2026], index=2, key='fin_y')
        f3, f4 = st.columns(2)
        if period_type == "Month": sel_spec = f3.selectbox("Month", range(1, 13), index=0, key='fin_m')
        elif period_type == "Quarter": sel_spec = f3.selectbox("Quarter", [1, 2, 3, 4], index=0, key='fin_q')
        else: sel_spec = 0

    start_date, end_date = get_date_filter_range(period_type, sel_year, sel_spec)
    
    # 1. CASH FLOW & P&L DATA
    inflow, cash_in = [], 0.0
    col_coll_amt = get_col_by_letter(df_coll, 'R')
    col_coll_y = get_col_by_letter(df_coll, 'Q')
    col_coll_m = get_col_by_letter(df_coll, 'P')
    
    if col_coll_amt:
        for _, row in df_coll.iterrows():
            try:
                y, m = int(clean_currency(row[col_coll_y])), int(clean_currency(row[col_coll_m]))
                valid = False
                if period_type=="Year" and y==sel_year: valid=True
                elif period_type=="Month" and y==sel_year and m==sel_spec: valid=True
                elif period_type=="Quarter":
                    if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: valid=True
                if valid:
                    amt = clean_currency(row[col_coll_amt])
                    inflow.append({"Amount": amt, "Category": "Revenue"})
                    cash_in += amt
            except: continue

    cash_out = 0.0
    col_exp_amt = get_col_by_letter(df_exp, 'X')
    col_exp_y = get_col_by_letter(df_exp, 'W')
    col_exp_m = get_col_by_letter(df_exp, 'V')
    
    if col_exp_amt:
        for _, row in df_exp.iterrows():
            try:
                y, m = int(clean_currency(row[col_exp_y])), int(clean_currency(row[col_exp_m]))
                valid = False
                if period_type=="Year" and y==sel_year: valid=True
                elif period_type=="Month" and y==sel_year and m==sel_spec: valid=True
                elif period_type=="Quarter":
                    if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: valid=True
                if valid: cash_out += clean_currency(row[col_exp_amt])
            except: continue

    # Car Expenses (Deductions vs Payouts)
    col_cexp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_cexp_y = get_col_by_letter(df_car_exp, 'Y')
    col_cexp_m = get_col_by_letter(df_car_exp, 'X')
    col_cexp_d = get_col_by_letter(df_car_exp, 'W')
    col_cexp_due_from = get_col_by_letter(df_car_exp, 'O') # "Due From" -> "Car Owner" means deduction
    col_cexp_car = get_col_by_letter(df_car_exp, 'S')
    col_cexp_item = get_col_by_letter(df_car_exp, 'F') # "Type of Expense" -> Rental/Payout means Payment TO Owner

    # Maps for Ledger
    deductions_in_period = {} # cid -> amount
    payments_to_owner_period = {} # cid -> amount
    payments_to_owner_lifetime = {} # cid -> amount

    if col_cexp_amt:
        for _, row in df_car_exp.iterrows():
            try:
                amt = clean_currency(row[col_cexp_amt])
                cid = clean_id_tag(row[col_cexp_car])
                y, m = int(clean_currency(row[col_cexp_y])), int(clean_currency(row[col_cexp_m]))
                
                # Check Period Validity
                is_in_period = False
                if period_type=="Year" and y==sel_year: is_in_period=True
                elif period_type=="Month" and y==sel_year and m==sel_spec: is_in_period=True
                elif period_type=="Quarter":
                    if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: is_in_period=True
                
                # Global Expense Sum
                if is_in_period: cash_out += amt

                # 1. Deductions (Due FROM Owner)
                # Check Col O for "Car Owner" or "صاحب السيارة"
                due_from = str(row[col_cexp_due_from]).lower()
                is_deduction = "owner" in due_from or "صاحب" in due_from or "المالك" in due_from
                
                if is_deduction:
                    if is_in_period: deductions_in_period[cid] = deductions_in_period.get(cid, 0) + amt
                
                # 2. Payments (Paid TO Owner)
                # Check Col F/I for "Rental" or "Rent" or "دفع"
                exp_type = str(row[col_cexp_item]).lower()
                is_payout = "rent" in exp_type or "ايجار" in exp_type or "owner" in exp_type
                
                if is_payout:
                    payments_to_owner_lifetime[cid] = payments_to_owner_lifetime.get(cid, 0) + amt
                    if is_in_period: payments_to_owner_period[cid] = payments_to_owner_period.get(cid, 0) + amt

            except: continue

    # 2. OWNER LEDGER CALCULATION
    owner_ledger = []
    total_owner_payouts_due = 0.0 # For P&L Chart

    # Car Columns
    col_code = get_col_by_letter(df_cars, 'A')
    col_status = get_col_by_letter(df_cars, 'AZ')
    col_contract_start = get_col_by_letter(df_cars, 'AW')
    col_monthly_fee = get_col_by_letter(df_cars, 'CJ')
    col_pay_freq = get_col_by_letter(df_cars, 'CK') # Every X Days
    col_deduct_pct = get_col_by_letter(df_cars, 'CL') # %
    col_brokerage = get_col_by_letter(df_cars, 'CM') # Extra Fee

    for _, car in df_cars.iterrows():
        try:
            # Active Check
            if col_status and not any(x in str(car[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
            cid = clean_id_tag(car[col_code])
            
            # Fee Logic
            base_fee = clean_currency(car[col_monthly_fee])
            freq_days = clean_currency(car[col_pay_freq])
            if freq_days == 0: freq_days = 30 # Default to monthly
            
            deduct_pct = clean_currency(car[col_deduct_pct])
            brokerage = clean_currency(car[col_brokerage])
            
            # Start Date
            s_date = pd.to_datetime(car[col_contract_start], errors='coerce')
            if pd.isna(s_date): s_date = datetime(2023, 1, 1)
            
            # --- A. Calculate "Due Date" for Current Period ---
            # If contract started on 15th, due date is 15th of the selected month
            try: due_day = datetime(sel_year, sel_spec if period_type=="Month" else 1, s_date.day)
            except: due_day = datetime(sel_year, sel_spec, 28) # Handle Feb 30 etc
            
            due_date_display = due_day.strftime("%Y-%m-%d") if period_type=="Month" else "Various"

            # --- B. Calculate Exact Amount Due (Period) ---
            # Formula: (Base Fee * (Days in Period / Freq Days)) - % Deduction + Brokerage
            
            days_in_view = 30 # Default month
            if period_type == "Quarter": days_in_view = 90
            elif period_type == "Year": days_in_view = 365
            
            # How many "Payment Cycles" fit in this view?
            cycles = days_in_view / freq_days
            
            gross_due = base_fee * cycles
            ops_fee_deduction = gross_due * (deduct_pct / 100)
            
            # Maint Deductions from Expenses Sheet
            maint_deduction = deductions_in_period.get(cid, 0)
            
            net_due_period = gross_due - ops_fee_deduction + brokerage - maint_deduction
            total_owner_payouts_due += net_due_period

            # --- C. Historical Balance ---
            # Total Days Active since contract start until NOW
            days_active = (datetime.now() - s_date).days
            if days_active < 0: days_active = 0
            
            total_cycles_lifetime = days_active / freq_days
            lifetime_gross = base_fee * total_cycles_lifetime
            lifetime_ops_fee = lifetime_gross * (deduct_pct / 100)
            
            # Approx lifetime brokerage (assuming monthly)
            lifetime_brokerage = brokerage * (days_active / 30)
            
            # We don't have "Lifetime Maintenance Deductions" easily unless we scan all years. 
            # For now, let's use the Payments vs Accrued Revenue
            
            lifetime_accrued = lifetime_gross - lifetime_ops_fee + lifetime_brokerage
            lifetime_paid = payments_to_owner_lifetime.get(cid, 0)
            
            balance = lifetime_accrued - lifetime_paid

            owner_ledger.append({
                "Car": f"{car[get_col_by_letter(df_cars, 'B')]} {car[get_col_by_letter(df_cars, 'E')]}",
                "Due Date": due_date_display,
                "Gross Fee": format_egp(gross_due),
                "Deductions": format_egp(maint_deduction + ops_fee_deduction),
                "Net Due": net_due_period, # Number for logic, formatted later
                "Paid": payments_to_owner_period.get(cid, 0),
                "Balance": balance
            })
        except: continue

    # TABS
    tab1, tab2, tab3 = st.tabs(["Cash", "P&L", "Ledger"])
    
    with tab1:
        net = cash_in - cash_out
        c1, c2 = st.columns(2)
        c1.metric("In", format_egp(cash_in))
        c2.metric("Out", format_egp(cash_out), delta_color="inverse")
        st.metric("Net", format_egp(net))
        fig = go.Figure(go.Waterfall(measure=["relative", "relative", "total"], x=["In", "Out", "Net"], y=[cash_in, -cash_out, 0]))
        fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rev = sum(x['Amount'] for x in inflow if x['Category'] == "Revenue")
        # Profit = Revenue - (Ops Expenses) - (Owner Payables)
        profit = rev - cash_out - total_owner_payouts_due
        mrg = (profit/rev*100) if rev>0 else 0
        c1, c2 = st.columns(2)
        c1.metric("Rev", format_egp(rev)); c2.metric("Net", format_egp(profit), f"{mrg:.1f}%")
        
        b1, b2 = st.columns(2)
        with b1: 
            fig = px.pie(names=["Ops", "Owners", "Profit"], values=[cash_out, total_owner_payouts_due, max(0, profit)], hole=0.5)
            fig.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with b2:
            fig = go.Figure(data=[go.Bar(name='Rev', x=['P&L'], y=[rev]), go.Bar(name='Cost', x=['P&L'], y=[cash_out+total_owner_payouts_due])])
            fig.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if owner_ledger:
            df_l = pd.DataFrame(owner_ledger)
            
            # Format numbers
            for c in ["Net Due", "Paid", "Balance"]: 
                df_l[c] = df_l[c].apply(format_egp)
            
            # Conditional Formatting
            def highlight_balance(val):
                color = 'white'
                if 'M' in val or 'k' in val or ',' in val:
                    num = float(val.replace('M','000000').replace('k','000').replace(',',''))
                    if num > 100: color = '#ff4b4b' # Red (Owe money)
                    elif num < -100: color = '#00c853' # Green (Paid ahead)
                return f'color: {color}'

            st.dataframe(df_l.style.map(highlight_balance, subset=['Balance']), use_container_width=True, height=500)
        else: st.info("No Active Contracts")

# --- 9. MODULE 5: RISK RADAR (3-TIER BUCKETS & INSURANCE LOGIC) ---
def show_risk_radar(dfs):
    st.title("⚠️ Risk Radar")
    if not dfs: return
    
    df_cars = dfs['cars']
    today = datetime.now()
    
    # 0-3m (0-90), 3-6m (90-180), 6-12m (180-365)
    
    risks = {'License': [], 'Insurance': [], 'Contract': []}
    
    col_lic_end = get_col_by_letter(df_cars, 'AQ') 
    col_exam_end = get_col_by_letter(df_cars, 'BD')
    col_lic_status = get_col_by_letter(df_cars, 'AT')
    
    col_ins_end = get_col_by_letter(df_cars, 'BJ')
    col_ins_status = get_col_by_letter(df_cars, 'BN')
    
    col_con_end = get_col_by_letter(df_cars, 'BC')
    col_name = get_col_by_letter(df_cars, 'B')
    col_model = get_col_by_letter(df_cars, 'E')
    col_status = get_col_by_letter(df_cars, 'AZ')
    plate_cols = ['AC','AB','AA','Z','Y','X','W']

    for _, row in df_cars.iterrows():
        try:
            if col_status and not any(x in str(row[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
            cname = f"{row[col_name]} {row[col_model]}"
            plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])]).strip()
            
            # --- LICENSE (Dual) ---
            lic_valid = True
            if col_lic_status: lic_valid = any(x in str(row[col_lic_status]) for x in ['Valid', 'Active', 'ساري'])
            
            if lic_valid:
                d_lic = pd.to_datetime(row[col_lic_end], errors='coerce') if col_lic_end else None
                d_exam = pd.to_datetime(row[col_exam_end], errors='coerce') if col_exam_end else None
                target, reason = None, "License"
                
                if d_lic and d_exam:
                    if d_lic == d_exam: target, reason = d_lic, "License + Examination"
                    elif d_lic < d_exam: target, reason = d_lic, "License"
                    else: target, reason = d_exam, "Examination"
                elif d_lic: target, reason = d_lic, "License"
                elif d_exam: target, reason = d_exam, "Examination"
                
                if target:
                    days = (target - today).days
                    bucket = None
                    if days <= 90: bucket = "Critical (0-3 Months)"
                    elif days <= 180: bucket = "Warning (3-6 Months)"
                    elif days > 180: bucket = "Watchlist (6-12 Months)"
                    if bucket: risks['License'].append({'Car': cname, 'Plate': plate, 'Type': reason, 'Due': target.strftime("%Y-%m-%d"), 'Bucket': bucket, 'Days': days})

            # --- INSURANCE (Check if Exists) ---
            has_ins = False
            if col_ins_status:
                s_val = str(row[col_ins_status]).lower()
                if "yes" in s_val or "يوجد" in s_val: has_ins = True
            
            if has_ins and col_ins_end:
                d = pd.to_datetime(row[col_ins_end], errors='coerce')
                if pd.notnull(d):
                    days = (d - today).days
                    bucket = None
                    if days <= 90: bucket = "Critical (0-3 Months)"
                    elif days <= 180: bucket = "Warning (3-6 Months)"
                    elif days > 180: bucket = "Watchlist (6-12 Months)"
                    if bucket: risks['Insurance'].append({'Car': cname, 'Plate': plate, 'Due': d.strftime("%Y-%m-%d"), 'Bucket': bucket, 'Days': days})

            # --- CONTRACT ---
            if col_con_end:
                d = pd.to_datetime(row[col_con_end], errors='coerce')
                if pd.notnull(d):
                    days = (d - today).days
                    bucket = None
                    if days <= 90: bucket = "Critical (0-3 Months)"
                    elif days <= 180: bucket = "Warning (3-6 Months)"
                    elif days > 180: bucket = "Watchlist (6-12 Months)"
                    if bucket: risks['Contract'].append({'Car': cname, 'Plate': plate, 'Due': d.strftime("%Y-%m-%d"), 'Bucket': bucket, 'Days': days})

        except: continue

    t1, t2, t3 = st.tabs(["📄 License", "🛡️ Insurance", "📝 Contract"])
    
    def render_tab(category):
        items = risks[category]
        if not items:
            st.success("✅ No upcoming risks.")
            return
            
        df = pd.DataFrame(items).sort_values('Days')
        
        b1 = df[df['Bucket'] == "Critical (0-3 Months)"]
        b2 = df[df['Bucket'] == "Warning (3-6 Months)"]
        b3 = df[df['Bucket'] == "Watchlist (6-12 Months)"]
        
        with st.expander(f"🚨 Critical (0-3 Months) [{len(b1)}]", expanded=True):
            if not b1.empty: st.dataframe(b1.drop(columns=['Bucket', 'Days']), use_container_width=True)
            else: st.info("None")
            
        with st.expander(f"⚠️ Warning (3-6 Months) [{len(b2)}]", expanded=False):
            if not b2.empty: st.dataframe(b2.drop(columns=['Bucket', 'Days']), use_container_width=True)
            else: st.info("None")
            
        with st.expander(f"👀 Watchlist (6-12 Months) [{len(b3)}]", expanded=False):
            if not b3.empty: st.dataframe(b3.drop(columns=['Bucket', 'Days']), use_container_width=True)
            else: st.info("None")

    with t1: render_tab('License')
    with t2: render_tab('Insurance')
    with t3: render_tab('Contract')

# --- 10. NAV ---
st.sidebar.title("🚘 Rental OS")
page = st.sidebar.radio("", ["Operations", "Vehicle 360", "CRM", "Financial HQ", "Risk Radar"])
st.sidebar.markdown("---")
dfs = load_data_v3()
if page == "Operations": show_operations(dfs)
elif page == "Vehicle 360": show_vehicle_360(dfs)
elif page == "CRM": show_crm(dfs)
elif page == "Financial HQ": show_financial_hq(dfs)
elif page == "Risk Radar": show_risk_radar(dfs)
