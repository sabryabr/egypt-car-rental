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

# --- 3. DATA ENGINE (COMPLETE) ---
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
                
                # Sanitize Headers
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
                
                # Strict Row Enforcement
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
        'cars': "1LGvYd5qBVRobzyLuBqwDInjdINmQTxih", # Updated ID
        'orders': "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A",
        'clients': "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg",
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

# --- 5. MODULE 1: OPERATIONS ---
def show_operations(dfs):
    st.title("🏠 Operations Command Center")
    if dfs:
        df_orders = dfs['orders']
        df_cars = dfs['cars']
        
        car_map = {} 
        active_fleet_count = 0
        col_code = get_col_by_letter(df_cars, 'A')
        col_type = get_col_by_letter(df_cars, 'B')
        col_model = get_col_by_letter(df_cars, 'E')
        col_year = get_col_by_letter(df_cars, 'H')
        col_status = get_col_by_letter(df_cars, 'AZ')

        if col_code and col_status:
            active_cars = df_cars[df_cars[col_status].astype(str).str.contains('Valid|ساري', case=False, na=False)]
            active_fleet_count = len(active_cars)
            for _, row in df_cars.iterrows(): 
                try:
                    c_id = clean_id_tag(row[col_code])
                    c_label = f"{row[col_type]} {row[col_model]} ({row[col_year]})"
                    car_map[c_id] = c_label
                except: continue

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
                    
                    raw_car_id = row[col_car_ord]
                    car_id_clean = clean_id_tag(raw_car_id)
                    car_name = car_map.get(car_id_clean, f"Unknown ({raw_car_id})")

                    if s_date <= today <= e_date: active_rentals += 1
                    if e_date.date() == today.date(): returning_today += 1
                    if s_date.date() == today.date(): departing_today += 1
                    
                    timeline_data.append({
                        'Car': car_name, 'Start': s_date, 'End': e_date,
                        'Client': str(row[col_client]) if col_client else "N/A",
                        'Status': 'Active' if s_date <= today <= e_date else 'Scheduled'
                    })
                except: continue

        utilization = int((active_rentals / active_fleet_count * 100)) if active_fleet_count > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚗 Active Rentals", active_rentals, "Live")
        c2.metric("🔄 Returning Today", returning_today, delta_color="inverse")
        c3.metric("🛫 Departing Today", departing_today)
        c4.metric("📊 Utilization", f"{utilization}%", f"{active_fleet_count} Active Cars")
        
        st.divider()
        st.subheader("📅 Live Fleet Schedule")
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            df_timeline = df_timeline[df_timeline['Start'] > '2023-01-01']
            fig = px.timeline(df_timeline, x_start="Start", x_end="End", y="Car", color="Status",
                color_discrete_map={"Active": "#00C853", "Scheduled": "#2962FF"}, hover_data=["Client"])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=600, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white"))
            fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No bookings found.")

# --- 6. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 Vehicle 360° Profile")
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Vehicle Filter")
    car_options = {}
    col_code = get_col_by_letter(df_cars, 'A')
    
    if col_code:
        for _, row in df_cars.iterrows():
            try:
                c_id = clean_id_tag(row[col_code])
                c_label = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                car_options[c_label + f" ({c_id})"] = c_id
            except: continue

    selected_car_label = st.sidebar.selectbox("Select Vehicle", list(car_options.keys()))
    selected_car_id = car_options[selected_car_label]
    sel_year = st.sidebar.selectbox("Analysis Year", [2024, 2025, 2026], index=2)
    sel_month = st.sidebar.selectbox("Analysis Month", range(1, 13), index=0)

    car_row = df_cars[df_cars[col_code].astype(str).str.strip().str.replace(" ", "").str.lower() == selected_car_id].iloc[0]
    base_fee = clean_currency(car_row[get_col_by_letter(df_cars, 'CJ')])
    deduct_pct = clean_currency(car_row[get_col_by_letter(df_cars, 'CL')])
    
    month_revenue = 0.0
    trip_count = 0
    col_ord_start = get_col_by_letter(df_orders, 'L')
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_car = get_col_by_letter(df_orders, 'C')

    if col_ord_start:
        for _, row in df_orders.iterrows():
            if clean_id_tag(row[col_ord_car]) == selected_car_id:
                d = pd.to_datetime(row[col_ord_start], errors='coerce')
                if pd.notnull(d) and d.year == sel_year and d.month == sel_month:
                    month_revenue += clean_currency(row[col_ord_cost])
                    trip_count += 1

    month_expenses = 0.0
    col_exp_car = get_col_by_letter(df_car_exp, 'S')
    col_exp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_exp_m = get_col_by_letter(df_car_exp, 'X')
    col_exp_y = get_col_by_letter(df_car_exp, 'Y')

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            if clean_id_tag(row[col_exp_car]) == selected_car_id:
                try:
                    if int(clean_currency(row[col_exp_y])) == sel_year and int(clean_currency(row[col_exp_m])) == sel_month:
                        month_expenses += clean_currency(row[col_exp_amt])
                except: continue

    est_owner_cost = base_fee * (1 - (deduct_pct/100))
    net_profit = month_revenue - month_expenses - est_owner_cost

    with st.container():
        c1, c2, c3 = st.columns([1, 3, 2])
        with c2:
            st.subheader(f"{selected_car_label}")
        with c3:
            st.metric("Net Profit (Est.)", f"{net_profit:,.0f} EGP", delta=f"Rev: {month_revenue:,.0f}")

    st.divider()
    tab1, tab2 = st.tabs(["💰 P&L Analysis", "🛠️ Maintenance Log"])

    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenue", f"{month_revenue:,.0f}")
        k2.metric("Expenses", f"{month_expenses:,.0f}")
        k3.metric("Owner Fee", f"{est_owner_cost:,.0f}")
        k4.metric("Trips", trip_count)
        
        fig = go.Figure(data=[
            go.Bar(name='Revenue', x=['Financials'], y=[month_revenue], marker_color='#2ecc71'),
            go.Bar(name='Expenses', x=['Financials'], y=[month_expenses], marker_color='#e74c3c'),
            go.Bar(name='Owner', x=['Financials'], y=[est_owner_cost], marker_color='#f1c40f')
        ])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        exp_list = []
        if col_exp_car:
            col_exp_desc = get_col_by_letter(df_car_exp, 'I')
            for _, row in df_car_exp.iterrows():
                if clean_id_tag(row[col_exp_car]) == selected_car_id:
                    exp_list.append({"Date": f"{row[col_exp_y]}-{row[col_exp_m]}", "Item": row[col_exp_desc], "Cost": clean_currency(row[col_exp_amt])})
        if exp_list: st.dataframe(pd.DataFrame(exp_list), use_container_width=True)
        else: st.info("No expenses.")

# --- 7. MODULE 3: FINANCIAL HQ ---
def show_financial_hq(dfs):
    st.title("💰 Financial HQ (The CFO View)")
    if not dfs: return

    df_coll = dfs['collections']
    df_exp = dfs['expenses']
    df_car_exp = dfs['car_expenses']
    df_orders = dfs['orders']
    df_cars = dfs['cars']

    st.sidebar.markdown("---")
    st.sidebar.header("🗓️ Financial Period")
    sel_year = st.sidebar.selectbox("Fiscal Year", [2024, 2025, 2026], index=2)
    sel_month = st.sidebar.selectbox("Fiscal Month", range(1, 13), index=0)

    # 1. Process INFLOW
    inflow_data = []
    total_cash_in = 0.0
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

    # 2. Process OUTFLOW
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

    tab1, tab2, tab3 = st.tabs(["🌊 Cash Flow", "📉 P&L", "🤝 Owner Ledger"])

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
        real_profit = real_revenue - total_cash_out
        k1, k2 = st.columns(2)
        k1.metric("Real Revenue", format_egp(real_revenue))
        k2.metric("Net Operating Profit", format_egp(real_profit))

    with tab3:
        st.subheader(f"Owner Payouts: {sel_month}/{sel_year}")
        owner_ledger = []
        col_car_code = get_col_by_letter(df_cars, 'A')
        for _, car in df_cars.iterrows():
            try:
                car_id = clean_id_tag(car[col_car_code])
                base = clean_currency(car[get_col_by_letter(df_cars, 'CJ')])
                freq = clean_currency(car[get_col_by_letter(df_cars, 'CK')])
                deduct_pct = clean_currency(car[get_col_by_letter(df_cars, 'CL')])
                
                if base == 0: continue 
                if freq == 0: freq = 30
                
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
        else: st.info("No owner contracts found.")

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
