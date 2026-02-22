import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import string
import re
import time
from datetime import datetime, timedelta
import calendar
import pytz

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="Egypt Rental ERP 3.0", layout="wide", page_icon="🏢", initial_sidebar_state="expanded")

EGYPT_TZ = pytz.timezone('Africa/Cairo')
def get_now():
    """Forces all time calculations to Egypt local time."""
    return datetime.now(EGYPT_TZ).replace(tzinfo=None)

# --- 2. ENTERPRISE UI/CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
    
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #0e1117; color: #f8f9fa; text-align: right; }
    
    /* SaaS Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e2530 0%, #262c36 100%);
        border: 1px solid #3b4252;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white; text-align: right;
    }
    label[data-testid="stMetricLabel"] { font-size: 0.9rem !important; font-weight: 600; color: #8f9bba !important; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800; color: #50fa7b !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem !important; }

    /* Modern Tables */
    .stDataFrame { direction: ltr; width: 100%; text-align: left; }
    .stDataFrame div[data-testid="stHorizontalBlock"] { width: 100%; }
    th { text-align: left !important; font-family: 'Cairo'; background-color: #1e2530 !important; color: #8f9bba !important; font-weight: 600 !important; }
    td { text-align: left !important; font-family: 'Cairo'; border-bottom: 1px solid #3b4252 !important; }
    
    /* Tabs & Headers */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; margin-bottom: 1.5rem; flex-wrap: wrap; direction: rtl; border-bottom: 2px solid #3b4252; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 25px; font-size: 1.1rem; font-family: 'Cairo'; font-weight: 600; background: transparent; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #50fa7b !important; color: #50fa7b !important; }
    h1, h2, h3, h4, h5 { font-family: 'Cairo', sans-serif; text-align: right; color: #eceff4; font-weight: 800; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1f29; border-left: 1px solid #2e3440; font-family: 'Cairo'; direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# --- 3. BULLETPROOF DATA ENGINE ---
@st.cache_data(ttl=300)
def load_data_v3():
    if "gcp_service_account" not in st.secrets:
        st.error("⚠️ خطأ: يرجى إضافة بيانات الاعتماد (Secrets).")
        return None

    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)

    def fetch_sheet(sheet_id, range_name, header_row=0):
        for attempt in range(3):
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
                        if len(row_fixed) < target_len: row_fixed += [None] * (target_len - len(row_fixed))
                        clean_data.append(row_fixed)
                    return pd.DataFrame(clean_data, columns=clean_headers)
                return pd.DataFrame()
            except HttpError as e:
                if e.resp.status in [429, 500, 503]: time.sleep(2 ** attempt); continue
                else: return pd.DataFrame()
            except Exception: return pd.DataFrame()
        return pd.DataFrame()

    IDS = {
        'cars': "1tQVkPj7tCnrKsHEIs04a1WzzC04jpOWuLsXgXOkVMkk",
        'orders': "1T6j2xnRBTY31crQcJHioKurs4Rvaj-VlEQkm6joGxGM",
        'clients': "13YZOGdRCEy7IMZHiTmjLFyO417P8dD0m5Sh9xwKI8js",
        'expenses': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_expenses': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'collections': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg"
    }

    with st.spinner("🔄 جاري مزامنة قاعدة البيانات المركزية..."):
        dfs = {k: fetch_sheet(v, "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0) if k != 'orders' else fetch_sheet(v, "'صفحة الإدخالات للإيجارات'!A:ZZ", 1) for k, v in IDS.items()}
        return dfs

# --- 4. ROBUST HELPERS ---
def get_col_by_letter(df, letter):
    def letter_to_index(col_str):
        num = 0
        for c in col_str:
            if c.upper() in string.ascii_uppercase: num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num - 1
    idx = letter_to_index(letter)
    if idx < len(df.columns): return df.columns[idx]
    return None

def val(row, col_name):
    """Safely extracts a value from a row even if the column is missing."""
    if col_name is None: return None
    return row.get(col_name)

def clean_id_tag(x):
    if pd.isna(x): return "unknown"
    return str(x).strip().replace(" ", "").lower()

def clean_client_code(x):
    if pd.isna(x): return "unknown"
    s = str(x).strip()
    if s.endswith(".0"): s = s[:-2]
    return s

def clean_currency(x):
    if pd.isna(x) or x is None: return 0.0
    s = str(x).replace(',', '').replace('%', '').strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

def format_egp(x): return f"{x:,.0f} ج.م"
def format_usd(x): return f"${x:,.0f}"
def format_eur(x): return f"€{x:,.0f}"

def parse_ar_date(x):
    if pd.isna(x) or x is None or str(x).strip() == "": return pd.NaT
    s = str(x).strip().replace("صباحًا", "AM").replace("مساءً", "PM").replace("ص", "AM").replace("م", "PM")
    try: return pd.to_datetime(s)
    except: return pd.NaT

def get_status_badge(s_date, e_date):
    if pd.isna(s_date) or pd.isna(e_date): return "⚪ غير محدد"
    today = get_now()
    if s_date <= today <= e_date: return "🟢 نشط"
    elif s_date > today: return "🟡 قادم"
    else: return "⚪ مكتمل"

def get_date_filter_range(period_type, year, specifier):
    if period_type == "سنة": return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)
    elif period_type == "ربع سنوي":
        q_map = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
        s_m, e_m = q_map[specifier]
        _, last_day = calendar.monthrange(year, e_m)
        return datetime(year, s_m, 1), datetime(year, e_m, last_day, 23, 59, 59)
    else: 
        _, last_day = calendar.monthrange(year, specifier)
        return datetime(year, specifier, 1), datetime(year, specifier, last_day, 23, 59, 59)

# --- MODULE 1: CONTROL TOWER ---
def show_control_tower(dfs):
    st.title("🛰️ برج المراقبة (Control Tower)")
    if not dfs: return
    df_orders, df_cars = dfs['orders'], dfs['cars']

    today = get_now()
    active_rentals = 0
    checkins_today = 0
    checkouts_today = 0
    car_status_map = {} 
    
    col_start = get_col_by_letter(df_orders, 'L') 
    col_end = get_col_by_letter(df_orders, 'T')   
    col_car_ord = get_col_by_letter(df_orders, 'D') 
    
    if col_start and col_car_ord:
        for _, row in df_orders.iterrows():
            try:
                cid = clean_id_tag(val(row, col_car_ord))
                s = parse_ar_date(val(row, col_start))
                e = parse_ar_date(val(row, col_end))
                if pd.notnull(s) and pd.notnull(e):
                    if s <= today <= e: car_status_map[cid] = "🔴" 
                    if s.date() == today.date(): checkouts_today += 1 
                    if e.date() == today.date(): checkins_today += 1  
            except: continue

    car_map = {} 
    active_fleet_count = 0
    
    col_code, col_status, col_brand, col_model = get_col_by_letter(df_cars, 'A'), get_col_by_letter(df_cars, 'AZ'), get_col_by_letter(df_cars, 'B'), get_col_by_letter(df_cars, 'E')
    plate_cols = ['W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC']

    if col_code and col_status:
        valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
        cars_subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        active_fleet_count = len(cars_subset)
        
        for _, row in cars_subset.iterrows(): 
            try:
                c_id = clean_id_tag(val(row, col_code))
                c_name = f"{val(row, col_brand)} {val(row, col_model)}"
                plate = "".join([str(val(row, p)) + " " for p in plate_cols if pd.notnull(val(row, p))])
                indicator = car_status_map.get(c_id, "🟢") 
                if indicator == "🔴": active_rentals += 1 
                car_map[c_id] = f"{c_name} | {plate.strip()}"
            except: continue

    # High-Level Metrics
    st.markdown("### 📡 نبذة عن اليوم")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("إيجارات حية الآن", active_rentals, f"من أصل {active_fleet_count} سيارة", delta_color="off")
    k2.metric("متاح للإيجار", active_fleet_count - active_rentals, "جاهز للتسليم", delta_color="normal")
    k3.metric("تسليمات اليوم (خروج)", checkouts_today, "سيارات تغادر اليوم", delta_color="off")
    k4.metric("استلامات اليوم (عودة)", checkins_today, "سيارات تعود اليوم", delta_color="off")

    st.divider()

    # Timeline Generation (Next 30 Days)
    st.markdown("### 🗓️ الجدول الزمني (الـ 30 يوماً القادمة)")
    timeline_data = []
    col_client = get_col_by_letter(df_orders, 'C') 
    
    end_horizon = today + timedelta(days=30)
    start_horizon = today - timedelta(days=5) 
    
    if col_start and col_end and col_car_ord:
        for _, row in df_orders.iterrows():
            try:
                s_date = parse_ar_date(val(row, col_start))
                e_date = parse_ar_date(val(row, col_end))
                if pd.isna(s_date) or pd.isna(e_date): continue
                if not (s_date <= end_horizon and e_date >= start_horizon): continue
                
                car_id_clean = clean_id_tag(val(row, col_car_ord))
                if car_id_clean not in car_map: continue
                
                status = "نشط" if s_date <= today <= e_date else ("قادم" if s_date > today else "مكتمل")
                client_name = str(val(row, col_client)) if pd.notnull(val(row, col_client)) else "غير معروف"
                
                timeline_data.append({
                    'السيارة': car_map[car_id_clean], 'البدء': s_date, 'الانتهاء': e_date, 
                    'العميل': client_name, 'الحالة': status
                })
            except: continue

    df_timeline = pd.DataFrame(timeline_data) if timeline_data else pd.DataFrame(columns=['السيارة', 'البدء', 'الانتهاء', 'الحالة', 'العميل'])
    
    for car_name in sorted(list(car_map.values())):
        if car_name not in df_timeline['السيارة'].values:
            df_timeline = pd.concat([df_timeline, pd.DataFrame([{'السيارة': car_name, 'البدء': pd.NaT, 'الانتهاء': pd.NaT, 'الحالة': 'متاح', 'العميل': '-'}])], ignore_index=True)

    if not df_timeline.empty:
        color_map = {"نشط": "#ff4b4b", "قادم": "#9b59b6", "مكتمل": "#95a5a6", "متاح": "#2e3440"}
        fig = px.timeline(df_timeline, x_start="البدء", x_end="الانتهاء", y="السيارة", color="الحالة", color_discrete_map=color_map, hover_data=["العميل"])
        fig.update_yaxes(autorange="reversed", categoryorder='array', categoryarray=sorted(list(car_map.values())), type='category')
        fig.update_layout(height=max(400, len(car_map) * 35), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white", size=11), margin=dict(l=10, r=10, t=10, b=10))
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="solid", line_color="#50fa7b", annotation_text="اليوم")
        st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: MASTER ORDER BOOK ---
def show_order_book(dfs):
    st.title("📋 دفتر الطلبات الموحد")
    if not dfs: return
    df_orders = dfs['orders']

    st.markdown("سجل مركزي لجميع الحجوزات والعمليات مع تتبع الحالات المالية.")
    
    c_id = get_col_by_letter(df_orders, 'A')
    c_client = get_col_by_letter(df_orders, 'C')
    c_car = get_col_by_letter(df_orders, 'E')
    c_start = get_col_by_letter(df_orders, 'L')
    c_end = get_col_by_letter(df_orders, 'T')
    c_total = get_col_by_letter(df_orders, 'AU')
    c_dep_held = get_col_by_letter(df_orders, 'AW')
    c_egp = get_col_by_letter(df_orders, 'AX')
    c_usd = get_col_by_letter(df_orders, 'AY')
    c_eur = get_col_by_letter(df_orders, 'AZ')

    orders_list = []
    if c_id and c_start:
        for _, row in df_orders.iterrows():
            try:
                row_id = val(row, c_id)
                if pd.isna(row_id) or str(row_id).strip() == "": continue
                s, e = parse_ar_date(val(row, c_start)), parse_ar_date(val(row, c_end))
                stat = get_status_badge(s, e)
                
                orders_list.append({
                    "رقم الطلب": str(row_id),
                    "الحالة": stat,
                    "العميل": str(val(row, c_client)) if pd.notnull(val(row, c_client)) else "",
                    "السيارة": str(val(row, c_car)) if pd.notnull(val(row, c_car)) else "",
                    "البدء": s.strftime("%Y-%m-%d %I:%M %p") if pd.notnull(s) else "-",
                    "الانتهاء": e.strftime("%Y-%m-%d %I:%M %p") if pd.notnull(e) else "-",
                    "الإجمالي (EGP)": clean_currency(val(row, c_total)),
                    "المدفوع (EGP)": clean_currency(val(row, c_egp)),
                    "المدفوع (USD)": clean_currency(val(row, c_usd)),
                    "المدفوع (EUR)": clean_currency(val(row, c_eur)),
                    "الوديعة المعلقة": clean_currency(val(row, c_dep_held))
                })
            except: continue

    df_display = pd.DataFrame(orders_list)
    if not df_display.empty:
        col1, col2 = st.columns([2, 1])
        search_q = col1.text_input("🔍 بحث برقم الطلب، العميل، أو السيارة")
        stat_filter = col2.selectbox("فلتر الحالة", ["الكل", "🟢 نشط", "🟡 قادم", "⚪ مكتمل"])
        
        if search_q: 
            df_display = df_display[df_display.apply(lambda r: r.astype(str).str.contains(search_q, case=False).any(), axis=1)]
        if stat_filter != "الكل": 
            df_display = df_display[df_display["الحالة"] == stat_filter]

        df_display["الإجمالي (EGP)"] = df_display["الإجمالي (EGP)"].apply(format_egp)
        df_display["المدفوع (EGP)"] = df_display["المدفوع (EGP)"].apply(format_egp)
        df_display["المدفوع (USD)"] = df_display["المدفوع (USD)"].apply(format_usd)
        df_display["المدفوع (EUR)"] = df_display["المدفوع (EUR)"].apply(format_eur)
        df_display["الوديعة المعلقة"] = df_display["الوديعة المعلقة"].apply(format_egp)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد طلبات مسجلة أو جاري تحديث الأعمدة.")

# --- MODULE 3: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 ملف السيارات (Vehicle 360)")
    if not dfs: return

    df_cars, df_orders, df_car_exp = dfs['cars'], dfs['orders'], dfs['car_expenses']

    with st.expander("🔎 التحكم", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1: fleet_cat = st.radio("التصنيف", ["النشطة", "الأرشيف", "الكل"], horizontal=True)
        with col2:
            car_options = {}
            col_code, col_status = get_col_by_letter(df_cars, 'A'), get_col_by_letter(df_cars, 'AZ')
            plate_cols = ['W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC']
            if col_code and col_status:
                valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
                if fleet_cat == "النشطة": subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                elif fleet_cat == "الأرشيف": subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                else: subset = valid_rows 
                for _, row in subset.iterrows():
                    try:
                        c_id = clean_id_tag(val(row, col_code))
                        c_label = f"{val(row, get_col_by_letter(df_cars, 'B'))} {val(row, get_col_by_letter(df_cars, 'E'))}"
                        plate = "".join([str(val(row, p)) + " " for p in plate_cols if pd.notnull(val(row, p))])
                        car_options[f"[{val(row, col_code)}] {c_label} | {plate.strip()}"] = c_id
                    except: continue
            select_all = st.checkbox("تحديد الكل")
            selected_labels = st.multiselect("المركبات", list(car_options.keys()), default=list(car_options.keys()) if select_all else [])
            selected_ids = [car_options[l] for l in selected_labels]

        st.markdown("---")
        tf1, tf2, tf3 = st.columns(3)
        period_type = tf1.selectbox("عرض", ["شهر", "ربع سنوي", "سنة"], key='v360_p')
        sel_year = tf2.selectbox("السنة", [2024, 2025, 2026], index=2, key='v360_y')
        sel_spec = tf3.selectbox("الشهر/الربع", range(1, 13) if period_type == "شهر" else ([1, 2, 3, 4] if period_type == "ربع سنوي" else [0]), index=get_now().month-1 if period_type == "شهر" else 0)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)
    if not selected_ids: st.info("👈 اختر المركبات."); return

    trips_data, maint_list, exp_list = [], [], []
    total_revenue, total_maint, total_exp = 0.0, 0.0, 0.0
    
    col_ord_start, col_ord_end, col_ord_cost, col_ord_car, col_ord_id = get_col_by_letter(df_orders, 'L'), get_col_by_letter(df_orders, 'T'), get_col_by_letter(df_orders, 'AU'), get_col_by_letter(df_orders, 'D'), get_col_by_letter(df_orders, 'A')
    col_ord_loc_start, col_ord_loc_end, col_ord_dur_txt = get_col_by_letter(df_orders, 'M'), get_col_by_letter(df_orders, 'U'), get_col_by_letter(df_orders, 'V')

    if col_ord_start:
        for _, row in df_orders.iterrows():
            cid = clean_id_tag(val(row, col_ord_car))
            if cid in selected_ids:
                d_s, d_e = parse_ar_date(val(row, col_ord_start)), parse_ar_date(val(row, col_ord_end))
                if pd.notnull(d_s) and start_range <= d_s <= end_range:
                    rev = clean_currency(val(row, col_ord_cost))
                    total_revenue += rev
                    dur_txt = str(val(row, col_ord_dur_txt)) if pd.notnull(val(row, col_ord_dur_txt)) else ""
                    stat = get_status_badge(d_s, d_e)
                    
                    # Calculate simple daily rate fallback
                    days_calc = (d_e - d_s).days if pd.notnull(d_e) else 1
                    if days_calc == 0: days_calc = 1
                    
                    trips_data.append({
                        "السيارة": [k for k, v in car_options.items() if v == cid][0],
                        "الطلب": str(val(row, col_ord_id)),
                        "الحالة": stat,
                        "البدء": d_s.strftime('%Y-%m-%d %I:%M %p'),
                        "المدة": dur_txt,
                        "اليومية التقريبية": format_egp(rev / days_calc),
                        "الإجمالي": format_egp(rev)
                    })

    # Expenses
    col_exp_car, col_exp_amt, col_exp_y, col_exp_m, col_exp_d, col_exp_rec = get_col_by_letter(df_car_exp, 'S'), get_col_by_letter(df_car_exp, 'Z'), get_col_by_letter(df_car_exp, 'Y'), get_col_by_letter(df_car_exp, 'X'), get_col_by_letter(df_car_exp, 'W'), get_col_by_letter(df_car_exp, 'A')
    col_exp_type_ar, col_exp_maint_ar, col_exp_stmt_ar = get_col_by_letter(df_car_exp, 'E'), get_col_by_letter(df_car_exp, 'H'), get_col_by_letter(df_car_exp, 'K') 
    
    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            cid = clean_id_tag(val(row, col_exp_car))
            if cid in selected_ids:
                try:
                    y, m, d_val = int(clean_currency(val(row, col_exp_y))), int(clean_currency(val(row, col_exp_m))), int(clean_currency(val(row, col_exp_d)))
                    valid = (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec])
                    if valid:
                        amt = clean_currency(val(row, col_exp_amt))
                        type_str = str(val(row, col_exp_type_ar)).strip() 
                        
                        is_maint = ("صيانات" in type_str or "Maintenance" in type_str)
                        display_name = str(val(row, col_exp_maint_ar)) if is_maint else f"{type_str} - {str(val(row, col_exp_stmt_ar))}"

                        entry = {"السيارة": [k for k, v in car_options.items() if v == cid][0], "التاريخ": f"{y}-{m:02d}-{d_val:02d}", "البند": display_name, "التكلفة": format_egp(amt)}
                        if is_maint: maint_list.append(entry); total_maint += amt
                        else: exp_list.append(entry); total_exp += amt
                except: continue

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("إجمالي الإيرادات", format_egp(total_revenue))
    k2.metric("تكاليف صيانة", format_egp(total_maint), delta_color="inverse")
    k3.metric("مصروفات أخرى (وملاك)", format_egp(total_exp), delta_color="inverse")
    
    roi = total_revenue - total_maint - total_exp
    k4.metric("العائد الصافي (ROI)", format_egp(roi), delta_color="normal" if roi >= 0 else "inverse")
    
    t1, t2, t3 = st.tabs(["الرحلات", "الصيانة", "المصروفات"])
    with t1: st.dataframe(pd.DataFrame(trips_data), use_container_width=True) if trips_data else st.info("فارغ")
    with t2: st.dataframe(pd.DataFrame(maint_list), use_container_width=True) if maint_list else st.info("فارغ")
    with t3: st.dataframe(pd.DataFrame(exp_list), use_container_width=True) if exp_list else st.info("فارغ")

# --- MODULE 4: CRM ---
def show_crm(dfs):
    st.title("👥 إدارة العملاء")
    if not dfs: return
    df_orders, df_clients, df_cars = dfs['orders'], dfs['clients'], dfs['cars']

    car_display_map = {}
    col_code = get_col_by_letter(df_cars, 'A')
    plate_cols = ['W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC']
    if col_code:
        for _, row in df_cars.iterrows():
            try:
                cid = clean_id_tag(val(row, col_code))
                cname = f"{val(row, get_col_by_letter(df_cars, 'B'))} {val(row, get_col_by_letter(df_cars, 'E'))} | " + "".join([str(val(row, p)) + " " for p in plate_cols if pd.notnull(val(row, p))]).strip()
                car_display_map[cid] = cname
            except: continue

    client_id_map, client_db = {}, {}
    col_cl_id, col_cl_first, col_cl_last = get_col_by_letter(df_clients, 'A'), get_col_by_letter(df_clients, 'C'), get_col_by_letter(df_clients, 'D')
    if col_cl_id:
        for _, row in df_clients.iterrows():
            try:
                cid = clean_client_code(val(row, col_cl_id))
                fname = str(val(row, col_cl_first)) if pd.notnull(val(row, col_cl_first)) else ""
                lname = str(val(row, col_cl_last)) if pd.notnull(val(row, col_cl_last]) else ""
                full_name = f"{fname} {lname}".strip()
                if not full_name: continue
                client_id_map[cid] = full_name
                client_db[full_name] = {'Display': f"[{cid}] {full_name}", 'Name': full_name, 'Spend': 0, 'Trips': 0, 'History': [], 'DepositHeld': 0, 'PaidUSD': 0, 'PaidEUR': 0}
            except: continue

    col_ord_name, col_ord_cost, col_ord_s, col_ord_e = get_col_by_letter(df_orders, 'B'), get_col_by_letter(df_orders, 'AU'), get_col_by_letter(df_orders, 'L'), get_col_by_letter(df_orders, 'T')
    col_ord_dep_held, col_ord_usd, col_ord_eur = get_col_by_letter(df_orders, 'AW'), get_col_by_letter(df_orders, 'AY'), get_col_by_letter(df_orders, 'AZ')
    col_ord_car_name, col_ord_id = get_col_by_letter(df_orders, 'E'), get_col_by_letter(df_orders, 'A')

    if col_ord_name:
        for _, row in df_orders.iterrows():
            try:
                raw_val = clean_client_code(val(row, col_ord_name))
                if not raw_val or raw_val == "nan": continue
                real_name = client_id_map.get(raw_val, raw_val) 
                if real_name not in client_db: client_db[real_name] = {'Display': f"[?] {real_name}", 'Name': real_name, 'Spend': 0, 'Trips': 0, 'History': [], 'DepositHeld': 0, 'PaidUSD': 0, 'PaidEUR': 0}
                
                rec = client_db[real_name]
                amt, usd, eur, dep = clean_currency(val(row, col_ord_cost)), clean_currency(val(row, col_ord_usd)), clean_currency(val(row, col_ord_eur)), clean_currency(val(row, col_ord_dep_held))
                s, e = parse_ar_date(val(row, col_ord_s)), parse_ar_date(val(row, col_ord_e))
                
                stat = get_status_badge(s, e)
                
                rec['Spend'] += amt
                rec['PaidUSD'] += usd
                rec['PaidEUR'] += eur
                rec['DepositHeld'] += dep
                rec['Trips'] += 1
                rec['History'].append({
                    "رقم الطلب": str(val(row, col_ord_id)),
                    "السيارة": str(val(row, col_ord_car_name)),
                    "البدء": s.strftime("%Y-%m-%d") if pd.notnull(s) else "-", 
                    "التكلفة": format_egp(amt), "الحالة": stat, "وديعة معلقة": format_egp(dep)
                })
            except: continue

    df_crm = pd.DataFrame([{'Display': v['Display'], 'إنفاق (EGP)': format_egp(v['Spend']), 'إنفاق (USD)': format_usd(v['PaidUSD']), 'إنفاق (EUR)': format_eur(v['PaidEUR']), 'ودائع في ذمتنا': format_egp(v['DepositHeld']), 'رحلات': v['Trips'], 'Key': v['Name'], 'SpendRaw': v['Spend']} for v in client_db.values()])
    
    if not df_crm.empty:
        df_crm = df_crm.sort_values('SpendRaw', ascending=False)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("إجمالي العملاء المسجلين", len(client_db))
        c2.metric("أكبر عميل إنفاقاً", df_crm.iloc[0]['Display'].split("] ")[-1] if len(df_crm)>0 else "-")
        c3.metric("إجمالي الودائع المحتجزة للعملاء", format_egp(sum(v['DepositHeld'] for v in client_db.values())))

        st.divider()
        col_list, col_detail = st.columns([1, 2])
        with col_list:
            search = st.text_input("🔍 بحث عن عميل")
            if search: df_crm = df_crm[df_crm['Display'].str.contains(search, case=False, na=False)]
            selection = st.dataframe(df_crm[['Display', 'إنفاق (EGP)', 'ودائع في ذمتنا']], use_container_width=True, height=500, on_select="rerun", selection_mode="single-row", hide_index=True)
        with col_detail:
            if selection.selection.rows:
                client_data = client_db[df_crm.iloc[selection.selection.rows[0]]['Key']]
                st.info(f"**{client_data['Display']}**")
                
                tier = "🌟 VIP" if client_data['Spend'] > 50000 else "👤 Regular"
                st.markdown(f"**تصنيف العميل:** {tier}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("الإجمالي EGP", format_egp(client_data['Spend']))
                m2.metric("الإجمالي USD", format_usd(client_data['PaidUSD']))
                m3.metric("الإجمالي EUR", format_eur(client_data['PaidEUR']))
                m4.metric("وديعة معلقة", format_egp(client_data['DepositHeld']), delta_color="inverse" if client_data['DepositHeld']>0 else "off")
                
                if client_data['History']: st.dataframe(pd.DataFrame(client_data['History']), use_container_width=True, hide_index=True)
            else: st.info("👈 اختر عميلاً لرؤية محفظته وتاريخه.")
    else:
        st.info("لا توجد بيانات للعملاء. تأكد من إضافة عملاء وطلبات صحيحة في الشيت.")

# --- MODULE 5: FINANCIAL HQ ---
def show_financial_hq(dfs):
    st.title("💰 المركز المالي (Financial HQ)")
    if not dfs: return

    df_coll, df_exp, df_car_exp, df_cars, df_orders = dfs['collections'], dfs['expenses'], dfs['car_expenses'], dfs['cars'], dfs['orders']

    with st.expander("🗓️ إعدادات الفترة و الحساب", expanded=True):
        f1, f2, f3 = st.columns(3)
        period_type = f1.selectbox("نوع العرض", ["شهر", "ربع سنوي", "سنة"], key='fin_p')
        sel_year = f2.selectbox("السنة المالية", [2024, 2025, 2026, 2027], index=2, key='fin_y')
        calc_method = f3.selectbox("طريقة حساب الملاك", ["عن الفترة المحددة فقط", "تراكمي حتى نهاية الفترة"])
        f4, f5 = st.columns(2)
        if period_type == "شهر": sel_spec = f4.selectbox("الشهر", range(1, 13), index=get_now().month-1, key='fin_m')
        elif period_type == "ربع سنوي": sel_spec = f4.selectbox("الربع", [1, 2, 3, 4], index=0, key='fin_q')
        else: sel_spec = 0

    start_date, end_date = get_date_filter_range(period_type, sel_year, sel_spec)
    
    inflow_cats, expense_cats = {}, {}
    cash_in, cash_out = 0.0, 0.0
    
    # 1. Orders Data
    total_egp, total_usd, total_eur = 0.0, 0.0, 0.0
    deposits_collected, deposits_refunded, deposits_held = 0.0, 0.0, 0.0
    
    col_ord_s = get_col_by_letter(df_orders, 'L')
    col_ord_dep_coll, col_ord_dep_ref, col_ord_dep_held = get_col_by_letter(df_orders, 'AB'), get_col_by_letter(df_orders, 'AV'), get_col_by_letter(df_orders, 'AW')
    col_ord_egp, col_ord_usd, col_ord_eur = get_col_by_letter(df_orders, 'AX'), get_col_by_letter(df_orders, 'AY'), get_col_by_letter(df_orders, 'AZ')
    
    if col_ord_s:
        for _, row in df_orders.iterrows():
            try:
                s = parse_ar_date(val(row, col_ord_s))
                if pd.notnull(s) and start_date <= s <= end_date:
                    deposits_collected += clean_currency(val(row, col_ord_dep_coll))
                    deposits_refunded += clean_currency(val(row, col_ord_dep_ref))
                    deposits_held += clean_currency(val(row, col_ord_dep_held))
                    
                    total_egp += clean_currency(val(row, col_ord_egp))
                    total_usd += clean_currency(val(row, col_ord_usd))
                    total_eur += clean_currency(val(row, col_ord_eur))
            except: continue

    # 2. Collections (Legacy)
    col_coll_amt, col_coll_y, col_coll_m = get_col_by_letter(df_coll, 'R'), get_col_by_letter(df_coll, 'Q'), get_col_by_letter(df_coll, 'P')
    if col_coll_amt:
        for _, row in df_coll.iterrows():
            try:
                y, m = int(clean_currency(val(row, col_coll_y))), int(clean_currency(val(row, col_coll_m)))
                if (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]):
                    amt = clean_currency(val(row, col_coll_amt))
                    cash_in += amt
                    inflow_cats["تأجير عام"] = inflow_cats.get("تأجير عام", 0) + amt
            except: continue

    # 3. Expenses
    col_exp_amt, col_exp_y, col_exp_m, col_exp_type = get_col_by_letter(df_exp, 'X'), get_col_by_letter(df_exp, 'W'), get_col_by_letter(df_exp, 'V'), get_col_by_letter(df_exp, 'I')
    if col_exp_amt:
        for _, row in df_exp.iterrows():
            try:
                y, m = int(clean_currency(val(row, col_exp_y))), int(clean_currency(val(row, col_exp_m)))
                if (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]):
                    amt = clean_currency(val(row, col_exp_amt))
                    cash_out += amt
                    cat = str(val(row, col_exp_type)).strip() if pd.notnull(val(row, col_exp_type)) else "نثريات"
                    expense_cats[cat] = expense_cats.get(cat, 0) + amt
            except: continue

    # 4. Car Expenses
    col_cexp_amt, col_cexp_y, col_cexp_m, col_cexp_car, col_cexp_id_g = get_col_by_letter(df_car_exp, 'Z'), get_col_by_letter(df_car_exp, 'Y'), get_col_by_letter(df_car_exp, 'X'), get_col_by_letter(df_car_exp, 'S'), get_col_by_letter(df_car_exp, 'G') 
    ledger_history, contracting_audit = [], {}
    
    if col_cexp_amt:
        for _, row in df_car_exp.iterrows():
            try:
                amt, cid, type_id = clean_currency(val(row, col_cexp_amt)), clean_id_tag(val(row, col_cexp_car)), str(val(row, col_cexp_id_g)).strip()
                y, m = int(clean_currency(val(row, col_cexp_y))), int(clean_currency(val(row, col_cexp_m)))
                is_in_period = (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec])
                
                if is_in_period: 
                    cash_out += amt
                    cat_name = "صيانة / مخالفات" if type_id in ['3','4'] else ("دفعات تعاقد" if type_id == '1' else ("عمولات" if type_id == '8' else "مصروفات سيارة"))
                    expense_cats[cat_name] = expense_cats.get(cat_name, 0) + amt

                txn_date = datetime(y, m, 28)
                if type_id == '1': 
                    ledger_history.append({'CID': cid, 'Date': txn_date, 'Type': 'دفع تعاقد', 'Amount': -amt, 'Sort': 2, 'Icon': '⬇️🔴'})
                    if is_in_period:
                        if cid not in contracting_audit: contracting_audit[cid] = {'Due': 0, 'Paid': 0}
                        contracting_audit[cid]['Paid'] += amt
                elif type_id == '8': ledger_history.append({'CID': cid, 'Date': txn_date, 'Type': 'دفع عمولة', 'Amount': -amt, 'Sort': 2, 'Icon': '⬇️🔴'})
                elif type_id in ['3', '4']: ledger_history.append({'CID': cid, 'Date': txn_date, 'Type': 'خصم صيانة', 'Amount': -amt, 'Sort': 2, 'Icon': '🔧'})
            except: continue

    # 5. Generate Accruals
    col_code, col_owner_f, col_owner_l, col_contract_start, col_monthly_fee, col_pay_freq, col_deduct_pct, col_brokerage, col_model_yr, col_car_name, col_status = get_col_by_letter(df_cars, 'A'), get_col_by_letter(df_cars, 'BP'), get_col_by_letter(df_cars, 'BQ'), get_col_by_letter(df_cars, 'AW'), get_col_by_letter(df_cars, 'CJ'), get_col_by_letter(df_cars, 'CK'), get_col_by_letter(df_cars, 'CL'), get_col_by_letter(df_cars, 'CM'), get_col_by_letter(df_cars, 'H'), get_col_by_letter(df_cars, 'B'), get_col_by_letter(df_cars, 'AZ')
    plate_cols = ['W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC']
    cid_to_meta, future_limit = {}, datetime(sel_year, 12, 31) 

    if col_code:
        for _, row in df_cars.iterrows():
            try:
                if not any(x in str(val(row, col_status)).lower() for x in ['valid', 'active', 'ساري']): continue
                cid = clean_id_tag(val(row, col_code))
                owner_name = f"{val(row, col_owner_f)} {val(row, col_owner_l)}".strip()
                if not owner_name: owner_name = f"Owner {cid}"
                
                cname = f"{val(row, col_car_name)} {val(row, get_col_by_letter(df_cars, 'E'))}"
                plate = "".join([str(val(row, p)) + " " for p in plate_cols if pd.notnull(val(row, p))]).strip()
                yr = str(val(row, col_model_yr))
                
                search_key = f"[{cid}] {owner_name} - {cname} ({yr}) | {plate}"
                cid_to_meta[cid] = {'Label': search_key, 'Owner': owner_name, 'Car': cname}
                
                s_date = pd.to_datetime(val(row, col_contract_start), errors='coerce')
                if pd.isna(s_date): continue
                
                base_fee, freq_days, deduct_pct, brokerage = clean_currency(val(row, col_monthly_fee)), clean_currency(val(row, col_pay_freq)), clean_currency(val(row, col_deduct_pct)), clean_currency(val(row, col_brokerage))
                if freq_days == 0: freq_days = 30
                
                curr_date = s_date
                while curr_date <= future_limit:
                    net = base_fee * (1 - (deduct_pct/100)) + brokerage
                    ledger_history.append({'CID': cid, 'Date': curr_date, 'Type': 'استحقاق تعاقد', 'Amount': net, 'Sort': 1, 'Icon': '📄'})
                    if start_date <= curr_date <= end_date:
                        if cid not in contracting_audit: contracting_audit[cid] = {'Due': 0, 'Paid': 0}
                        contracting_audit[cid]['Due'] += net
                    curr_date += timedelta(days=freq_days)
            except: continue

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 P&L والخزينة", "💱 سلة العملات والودائع", "🤝 تسويات الملاك", "📋 تفاصيل التدقيق"])
    
    with tab1:
        st.markdown("##### الأرباح والخسائر للعمليات (Cash Flow)")
        rev = cash_in + total_egp 
        items = [('الإيرادات', rev)]
        for k, v in expense_cats.items(): items.append((k, -v))
        df_waterfall = pd.DataFrame(items, columns=['Category', 'Amount'])
        net_profit = df_waterfall['Amount'].sum()
        
        k1, k2, k3 = st.columns(3)
        k1.metric("إجمالي الدخل (EGP)", format_egp(rev))
        k2.metric("إجمالي المصروفات", format_egp(cash_out), delta_color="inverse")
        k3.metric("صافي الربح", format_egp(net_profit), delta_color="normal" if net_profit>=0 else "inverse")
        
        if not df_waterfall.empty:
            fig = go.Figure(go.Waterfall(
                name = "P&L", orientation = "v", measure = ["relative"] * len(df_waterfall) + ["total"],
                x = df_waterfall['Category'].tolist() + ["الصافي"], y = df_waterfall['Amount'].tolist() + [0],
                connector = {"line":{"color":"rgb(63, 63, 63)"}}, decreasing = {"marker":{"color":"#ef5350"}}, increasing = {"marker":{"color":"#50fa7b"}}, totals = {"marker":{"color":"#42a5f5"}}
            ))
            fig.update_layout(height=400, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("##### سلة العملات (حسب الطلبات في هذه الفترة)")
        v1, v2, v3 = st.columns(3)
        v1.metric("الخزينة (EGP)", format_egp(total_egp))
        v2.metric("الخزينة (USD)", format_usd(total_usd))
        v3.metric("الخزينة (EUR)", format_eur(total_eur))
        
        st.divider()
        st.markdown("##### دفتر الودائع والتأمين")
        d1, d2, d3 = st.columns(3)
        d1.metric("تأمين تم تحصيله", format_egp(deposits_collected)) 
        d2.metric("تأمين تم رده", format_egp(deposits_refunded))
        d3.metric("صافي معلق في ذمة الشركة", format_egp(deposits_held), delta_color="inverse" if deposits_held > 0 else "off")

    with tab3:
        df_all = pd.DataFrame(ledger_history)
        if not df_all.empty:
            df_all['Search_Label'] = df_all['CID'].map(lambda x: cid_to_meta.get(x, {}).get('Label', 'Unknown'))
            if calc_method == "عن الفترة المحددة فقط": df_filtered = df_all[(df_all['Date'] >= start_date) & (df_all['Date'] <= end_date)].copy()
            else: df_filtered = df_all[df_all['Date'] <= end_date].copy() 

            col_sel1, col_sel2 = st.columns([1, 3])
            with col_sel1: select_all = st.checkbox("تحديد الكل", value=True)
            all_options = sorted(df_filtered['Search_Label'].unique().tolist())
            
            with col_sel2:
                if select_all: selected_labels = all_options
                else: selected_labels = st.multiselect("اختر المالكين/السيارات", options=all_options)

            if selected_labels:
                final_data = df_filtered[df_filtered['Search_Label'].isin(selected_labels)].copy().sort_values(['Date', 'Sort'])
                total_due = final_data[final_data['Amount'] > 0]['Amount'].sum()
                total_paid = final_data[final_data['Amount'] < 0]['Amount'].sum()
                net_bal = total_due + total_paid
                
                m1, m2, m3 = st.columns(3)
                m1.metric("إجمالي المستحقات", format_egp(total_due))
                m2.metric("إجمالي المدفوعات والخصم", format_egp(abs(total_paid)))
                bal_icon = "⬇️🔴 لنا/مدفوع مقدماً" if net_bal < 0 else "⬆️🔴 للمالك/دين" 
                m3.metric("الرصيد النهائي", f"{bal_icon} {format_egp(abs(net_bal))}")
                
                st.divider()
                display = final_data[['Date', 'Search_Label', 'Icon', 'Type', 'Amount']].copy()
                display.columns = ['التاريخ', 'المركبة/المالك', ' ', 'البيان', 'القيمة']
                display['القيمة'] = display['القيمة'].apply(format_egp)
                display['التاريخ'] = display['التاريخ'].dt.strftime('%Y-%m-%d')
                st.dataframe(display, use_container_width=True)
            else: st.warning("يرجى اختيار مالك واحد على الأقل.")
        else: st.info("لا توجد بيانات مالية للملاك.")

    with tab4:
        st.subheader(f"📋 التدقيق الفردي للملاك ({period_type})")
        audit_data = []
        for cid, data in contracting_audit.items():
            diff = data['Paid'] - data['Due'] 
            status = "⚖️ متوازن"
            if diff > 100: status = "🟢 مدفوع بالزيادة"
            elif diff < -100: status = "🔴 معلق (عجز)"
            audit_data.append({
                "السيارة": cid_to_meta.get(cid, {}).get('Car', cid),
                "المستحق للفترة": format_egp(data['Due']), "المدفوع في الفترة": format_egp(data['Paid']),
                "الفارق": format_egp(diff), "الحالة": status
            })
        if audit_data: st.dataframe(pd.DataFrame(audit_data), use_container_width=True)
        else: st.info("لا توجد تعاقدات مستحقة في هذه الفترة.")

# --- MODULE 6: RISK RADAR ---
def show_risk_radar(dfs):
    st.title("⚠️ رادار المخاطر")
    if not dfs: return
    df_cars = dfs['cars']
    today = get_now() 
    risks = {'License': [], 'Insurance': [], 'Contract': []}
    
    col_lic_end, col_exam_end, col_lic_status = get_col_by_letter(df_cars, 'AQ'), get_col_by_letter(df_cars, 'BD'), get_col_by_letter(df_cars, 'AT')
    col_ins_end, col_ins_status, col_con_end = get_col_by_letter(df_cars, 'BJ'), get_col_by_letter(df_cars, 'BN'), get_col_by_letter(df_cars, 'BC')
    col_name, col_model, col_code, col_status = get_col_by_letter(df_cars, 'B'), get_col_by_letter(df_cars, 'E'), get_col_by_letter(df_cars, 'A'), get_col_by_letter(df_cars, 'AZ')
    plate_cols = ['W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC'] 

    if col_code:
        for _, row in df_cars.iterrows():
            try:
                if not any(x in str(val(row, col_status)).lower() for x in ['valid', 'active', 'ساري']): continue
                cid, cname = clean_id_tag(val(row, col_code)), f"[{clean_id_tag(val(row, col_code))}] {val(row, col_name)} {val(row, col_model)}"
                plate = "".join([str(val(row, p)) + " " for p in plate_cols if pd.notnull(val(row, p))]).strip()
                
                if col_lic_status and any(x in str(val(row, col_lic_status)).lower() for x in ['valid', 'active', 'ساري']):
                    d_lic, d_exam = pd.to_datetime(val(row, col_lic_end), errors='coerce'), pd.to_datetime(val(row, col_exam_end), errors='coerce')
                    target, reason = None, "ترخيص"
                    if pd.notnull(d_lic) and pd.notnull(d_exam): target, reason = (d_lic, "ترخيص + فحص") if d_lic == d_exam else ((d_lic, "ترخيص") if d_lic < d_exam else (d_exam, "فحص"))
                    elif pd.notnull(d_lic): target, reason = d_lic, "ترخيص"
                    elif pd.notnull(d_exam): target, reason = d_exam, "فحص"
                    if target:
                        days = (target - today).days
                        bucket = "🔴 خطر مرتفع (0-3 أشهر)" if days <= 90 else ("🟡 خطر متوسط (3-6 أشهر)" if days <= 180 else "🟢 خطر منخفض (> 6 أشهر)")
                        risks['License'].append({'السيارة': cname, 'اللوحة': plate, 'السبب': reason, 'الاستحقاق': target.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

                if col_ins_status and ("yes" in str(val(row, col_ins_status)).lower() or "يوجد" in str(val(row, col_ins_status)).lower()):
                    d = pd.to_datetime(val(row, col_ins_end), errors='coerce')
                    if pd.notnull(d):
                        days = (d - today).days
                        bucket = "🔴 خطر مرتفع (0-3 أشهر)" if days <= 90 else ("🟡 خطر متوسط (3-6 أشهر)" if days <= 180 else "🟢 خطر منخفض (> 6 أشهر)")
                        risks['Insurance'].append({'السيارة': cname, 'اللوحة': plate, 'الاستحقاق': d.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

                if col_con_end:
                    d = pd.to_datetime(val(row, col_con_end), errors='coerce')
                    if pd.notnull(d):
                        days = (d - today).days
                        bucket = "🔴 خطر مرتفع (0-3 أشهر)" if days <= 90 else ("🟡 خطر متوسط (3-6 أشهر)" if days <= 180 else "🟢 خطر منخفض (> 6 أشهر)")
                        risks['Contract'].append({'السيارة': cname, 'اللوحة': plate, 'الاستحقاق': d.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})
            except: continue

    t1, t2, t3 = st.tabs(["📄 الترخيص والفحص", "🛡️ التأمين", "📝 عقود الملاك"])
    def render_tab(category):
        if not risks[category]: st.success("✅ جميع التواريخ سليمة."); return
        df = pd.DataFrame(risks[category]).sort_values('Days')
        for label in ["🔴 خطر مرتفع (0-3 أشهر)", "🟡 خطر متوسط (3-6 أشهر)", "🟢 خطر منخفض (> 6 أشهر)"]:
            subset = df[df['التصنيف'] == label]
            with st.expander(f"{label} [{len(subset)}]", expanded=(label == "🔴 خطر مرتفع (0-3 أشهر)")):
                if not subset.empty: st.dataframe(subset.drop(columns=['التصنيف', 'Days']), use_container_width=True)
                else: st.info("لا يوجد")

    with t1: render_tab('License')
    with t2: render_tab('Insurance')
    with t3: render_tab('Contract')

# --- MAIN NAV ---
st.sidebar.title("🚘 Egypt Rental ERP")
page = st.sidebar.radio("", ["العمليات", "دفتر الطلبات الموحد", "ملف السيارات", "إدارة العملاء", "المركز المالي", "رادار المخاطر"])
st.sidebar.markdown("---")
dfs = load_data_v3()
if page == "العمليات": show_control_tower(dfs)
elif page == "دفتر الطلبات الموحد": show_order_book(dfs)
elif page == "ملف السيارات": show_vehicle_360(dfs)
elif page == "إدارة العملاء": show_crm(dfs)
elif page == "المركز المالي": show_financial_hq(dfs)
elif page == "رادار المخاطر": show_risk_radar(dfs)
