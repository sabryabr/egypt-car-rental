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
import calendar

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #1e2530; color: white; }
    div[data-testid="metric-container"] {
        background-color: #262730; border: 1px solid #464b5d; border-radius: 10px; padding: 15px; color: white;
    }
    label[data-testid="stMetricLabel"] { color: #b0b3b8 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    .stDataFrame { direction: ltr; }
    div[data-testid="stExpander"] { border: 1px solid #464b5d; border-radius: 8px; }
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
    st.title("🏠 Operations Command Center")
    if not dfs: return

    df_orders = dfs['orders']
    df_cars = dfs['cars']

    with st.expander("🔎 Filters & View Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        period_type = c1.selectbox("Period Type", ["Month", "Quarter", "Year"])
        sel_year = c2.selectbox("Year", [2024, 2025, 2026, 2027], index=2)
        
        if period_type == "Month":
            sel_spec = c3.selectbox("Month", range(1, 13), index=datetime.now().month-1)
        elif period_type == "Quarter":
            sel_spec = c3.selectbox("Quarter", [1, 2, 3, 4], index=0)
        else: sel_spec = 0 
            
        fleet_status = c4.selectbox("Fleet Filter", ["Active Only", "All Cars", "Inactive Only"], index=0)

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
                c_name = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]} ({row[get_col_by_letter(df_cars, 'H')]})"
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
    c1.metric("🚗 Active Rentals", active_rentals, "Live")
    c2.metric("📅 Future Bookings", future_orders, "Paid & Pending")
    c3.metric("🔄 Returning Today", returning_today, delta_color="inverse")
    c4.metric("📊 Utilization", f"{utilization:.1f}%", f"{active_fleet_count} Visible Cars")
    
    st.divider()
    st.subheader(f"📅 Fleet Schedule ({period_type})")
    
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
        fig.update_layout(height=max(500, len(all_car_names) * 50), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white"), xaxis=dict(showgrid=True, gridcolor="#333", range=[start_range, end_range]))
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("No active fleet found.")

# --- 6. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 Vehicle 360° Profile")
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    with st.expander("🔎 Vehicle Control Panel", expanded=True):
        col_filters_1, col_filters_2 = st.columns([1, 2])
        
        with col_filters_1:
            # UPDATED: Added "All Fleet" option
            fleet_cat = st.radio("Fleet Category", ["Active Fleet", "Inactive/History", "All Fleet (Active + Inactive)"], horizontal=True)
            
        with col_filters_2:
            car_options = {}
            col_code = get_col_by_letter(df_cars, 'A')
            col_status = get_col_by_letter(df_cars, 'AZ')
            plate_cols = ['AC','AB','AA','Z','Y','X','W']
            
            if col_code and col_status:
                valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
                
                # UPDATED: Logic for "All Fleet"
                if fleet_cat == "Active Fleet":
                    subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                elif fleet_cat == "Inactive/History":
                    subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                else:
                    subset = valid_rows # All Fleet

                for _, row in subset.iterrows():
                    try:
                        c_id = clean_id_tag(row[col_code])
                        c_label = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                        plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                        car_options[f"[{row[col_code]}] {c_label} | {plate.strip()}"] = c_id
                    except: continue

            selected_labels = st.multiselect("Select Vehicles to Compare", list(car_options.keys()))
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
            show_only_active = st.checkbox("Hide cars with no revenue in period", value=False)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    if not selected_ids:
        st.info("👈 Please select vehicles above to generate the report.")
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
        st.warning("No trips found for selected vehicles in this period.")
        return

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

# --- 7. MODULE 3: FINANCIAL HQ ---
def show_financial_hq(dfs):
    st.title("💰 Financial HQ (The CFO View)")
    if not dfs: return

    df_coll = dfs['collections']
    df_exp = dfs['expenses']
    df_car_exp = dfs['car_expenses']
    df_orders = dfs['orders']
    df_cars = dfs['cars']

    with st.expander("🗓️ Financial Period Settings", expanded=True):
        f1, f2 = st.columns(2)
        sel_year = f1.selectbox("Fiscal Year", [2024, 2025, 2026], index=2, key='fin_y')
        sel_month = f2.selectbox("Fiscal Month", range(1, 13), index=0, key='fin_m')

    # DATA PROCESSING
    inflow_data, total_cash_in = [], 0.0
    col_coll_amt = get_col_by_letter(df_coll, 'R')
    col_coll_order = get_col_by_letter(df_coll, 'L')
    col_coll_type = get_col_by_letter(df_coll, 'F')
    col_coll_m = get_col_by_letter(df_coll, 'P')
    col_coll_y = get_col_by_letter(df_coll, 'Q')

    order_dates = {}
    col_ord_id = get_col_by_letter(df_orders, 'A')
    col_ord_start = get_col_by_letter(df_orders, 'L')
    if col_ord_id:
        for _, row in df_orders.iterrows():
            oid = clean_id_tag(row[col_ord_id])
            s_date = pd.to_datetime(row[col_ord_start], errors='coerce')
            order_dates[oid] = {'start': s_date}

    if col_coll_amt:
        for _, row in df_coll.iterrows():
            try:
                if int(clean_currency(row[col_coll_y])) == sel_year and int(clean_currency(row[col_coll_m])) == sel_month:
                    amt = clean_currency(row[col_coll_amt])
                    ord_code = clean_id_tag(row[col_coll_order])
                    inc_type = str(row[col_coll_type]).lower()
                    category = "Realized Income"
                    
                    if "deposit" in inc_type or "تأمين" in inc_type: category = "Security Deposit (Liability)"
                    elif ord_code in order_dates:
                        dates = order_dates[ord_code]
                        if pd.notnull(dates['start']) and dates['start'] > datetime(sel_year, sel_month, 28):
                            category = "Deferred Revenue (Future)"
                    
                    inflow_data.append({'Amount': amt, 'Category': category})
                    total_cash_in += amt
            except: continue

    col_exp_amt = get_col_by_letter(df_exp, 'X')
    col_exp_m = get_col_by_letter(df_exp, 'V')
    col_exp_y = get_col_by_letter(df_exp, 'W')
    total_cash_out = 0.0
    
    if col_exp_amt:
        for _, row in df_exp.iterrows():
            try:
                if int(clean_currency(row[col_exp_y])) == sel_year and int(clean_currency(row[col_exp_m])) == sel_month:
                    total_cash_out += clean_currency(row[col_exp_amt])
            except: continue

    col_carexp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_carexp_m = get_col_by_letter(df_car_exp, 'X')
    col_carexp_y = get_col_by_letter(df_car_exp, 'Y')
    col_carexp_owner = get_col_by_letter(df_car_exp, 'O')
    owner_deductible_expenses = [] 
    
    if col_carexp_amt:
        for _, row in df_car_exp.iterrows():
            try:
                if int(clean_currency(row[col_carexp_y])) == sel_year and int(clean_currency(row[col_carexp_m])) == sel_month:
                    amt = clean_currency(row[col_carexp_amt])
                    total_cash_out += amt
                    if "owner" in str(row[col_carexp_owner]).lower() or "مالك" in str(row[col_carexp_owner]):
                        car_c = clean_id_tag(row[get_col_by_letter(df_car_exp, 'S')])
                        owner_deductible_expenses.append({'Car': car_c, 'Amount': amt})
            except: continue
    
    # Calculate Total Owner Payouts for P&L Chart
    total_owner_payouts = 0.0
    col_car_code = get_col_by_letter(df_cars, 'A')
    col_status = get_col_by_letter(df_cars, 'AZ')

    for _, car in df_cars.iterrows():
        try:
            if col_status and not any(x in str(car[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
            car_id = clean_id_tag(car[col_car_code])
            base = clean_currency(car[get_col_by_letter(df_cars, 'CJ')])
            deduct_pct = clean_currency(car[get_col_by_letter(df_cars, 'CL')])
            
            monthly_gross = (base / 30) * 30
            ops_fee = monthly_gross * (deduct_pct / 100)
            maint_deduction = sum(x['Amount'] for x in owner_deductible_expenses if x['Car'] == car_id)
            total_owner_payouts += (monthly_gross - ops_fee - maint_deduction)
        except: continue

    # TABS
    tab1, tab2, tab3 = st.tabs(["🌊 Cash Flow", "📉 P&L (Detailed)", "🤝 Owner Ledger"])

    with tab1:
        net_cash = total_cash_in - total_cash_out
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cash In", format_egp(total_cash_in), "Collections")
        c2.metric("Total Cash Out", format_egp(total_cash_out), "-Expenses", delta_color="inverse")
        c3.metric("Net Liquidity", format_egp(net_cash), "Available Cash")
        
        fig_water = go.Figure(go.Waterfall(measure = ["relative", "relative", "total"], x = ["In", "Out", "Net"], y = [total_cash_in, -total_cash_out, 0]))
        st.plotly_chart(fig_water, use_container_width=True)

    with tab2:
        real_revenue = sum(x['Amount'] for x in inflow_data if x['Category'] == "Realized Income")
        real_profit = real_revenue - total_cash_out - total_owner_payouts
        margin = (real_profit / real_revenue * 100) if real_revenue > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Real Revenue", format_egp(real_revenue))
        k2.metric("Op. Expenses", format_egp(total_cash_out), delta_color="inverse")
        k3.metric("Owner Payouts", format_egp(total_owner_payouts), delta_color="inverse")
        k4.metric("Net Profit", format_egp(real_profit), f"{margin:.1f}% Margin")

        # BREAKDOWN CHARTS
        b1, b2 = st.columns(2)
        with b1:
            # Expense Composition
            fig_pie = px.pie(names=["Op. Expenses", "Owner Payouts", "Net Profit"], 
                             values=[total_cash_out, total_owner_payouts, max(0, real_profit)],
                             title="Profit Distribution", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with b2:
             # Revenue vs Cost Bar
             fig_bar = go.Figure(data=[
                 go.Bar(name='Revenue', x=['P&L'], y=[real_revenue], marker_color='#2ecc71'),
                 go.Bar(name='Expenses', x=['P&L'], y=[total_cash_out + total_owner_payouts], marker_color='#e74c3c')
             ])
             fig_bar.update_layout(title="Revenue vs Costs", barmode='group')
             st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader(f"Owner Payouts: {sel_month}/{sel_year}")
        owner_ledger = []
        for _, car in df_cars.iterrows():
            try:
                if col_status and not any(x in str(car[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
                car_id = clean_id_tag(car[col_car_code])
                base = clean_currency(car[get_col_by_letter(df_cars, 'CJ')])
                deduct_pct = clean_currency(car[get_col_by_letter(df_cars, 'CL')])
                
                if base == 0: continue
                
                monthly_gross = (base / 30) * 30
                ops_fee = monthly_gross * (deduct_pct / 100)
                maint_deduction = sum(x['Amount'] for x in owner_deductible_expenses if x['Car'] == car_id)
                net_payout = monthly_gross - ops_fee - maint_deduction
                
                owner_ledger.append({
                    "Car": f"{car[get_col_by_letter(df_cars, 'B')]} {car[get_col_by_letter(df_cars, 'E')]}",
                    "Gross": monthly_gross, "Ops Fee": -ops_fee, "Maint": -maint_deduction, "Net": net_payout
                })
            except: continue
            
        if owner_ledger:
            df_ledger = pd.DataFrame(owner_ledger)
            for c in ["Gross", "Ops Fee", "Maint", "Net"]: df_ledger[c] = df_ledger[c].apply(format_egp)
            st.dataframe(df_ledger, use_container_width=True)
        else: st.info("No active owner contracts found.")

# --- 8. PAGE ROUTER ---
st.sidebar.title("🚘 Rental OS 3.0")
page = st.sidebar.radio("Navigate", ["🏠 Operations", "🚗 Vehicle 360", "👥 CRM", "💰 Financial HQ", "⚠️ Risk Radar"])
st.sidebar.markdown("---")

dfs = load_data_v3()

if page == "🏠 Operations": show_operations(dfs)
elif page == "🚗 Vehicle 360": show_vehicle_360(dfs)
elif page == "👥 CRM": st.title("👥 CRM (Coming Next)")
elif page == "💰 Financial HQ": show_financial_hq(dfs)
elif page == "⚠️ Risk Radar": st.title("⚠️ Risk Radar (Coming Next)")
