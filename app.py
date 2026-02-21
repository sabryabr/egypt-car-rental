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

# --- 1. إعداد التطبيق (CONFIG) ---
st.set_page_config(page_title="نظام إدارة التأجير 3.0", layout="wide", page_icon="🚘", initial_sidebar_state="auto")

EGYPT_TZ = pytz.timezone('Africa/Cairo')
def get_now():
    return datetime.now(EGYPT_TZ).replace(tzinfo=None)

# --- 2. تنسيق الواجهة (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    .main { direction: rtl; font-family: 'Cairo', sans-serif; background-color: #0e1117; color: white; text-align: right; }
    div[data-testid="metric-container"] {
        background-color: #262730; border: 1px solid #464b5d; border-radius: 8px; padding: 10px; 
        color: white; height: auto; min-height: 90px; overflow: hidden; text-align: right;
    }
    label[data-testid="stMetricLabel"] { font-size: 0.9rem !important; font-family: 'Cairo'; margin-bottom: 5px !important; color: #b0b3b8 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: bold; }
    .stDataFrame { direction: ltr; width: 100%; text-align: left; }
    .stDataFrame div[data-testid="stHorizontalBlock"] { width: 100%; }
    th { text-align: left !important; font-family: 'Cairo'; }
    td { text-align: left !important; font-family: 'Cairo'; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; margin-bottom: 1rem; flex-wrap: wrap; direction: rtl; }
    .stTabs [data-baseweb="tab"] { height: 45px; padding: 0 20px; font-size: 1rem; font-family: 'Cairo'; flex-grow: 1; }
    h1, h2, h3, h4, h5 { font-family: 'Cairo', sans-serif; text-align: right; }
    [data-testid="stSidebar"] { font-family: 'Cairo'; direction: rtl; text-align: right; }
    @media (max-width: 640px) { div[data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: 100px !important; } }
</style>
""", unsafe_allow_html=True)

# --- 3. محرك البيانات (DATA ENGINE) ---
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

    with st.spinner("🔄 جاري تحميل البيانات..."):
        dfs = {k: fetch_sheet(v, "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0) if k != 'orders' else fetch_sheet(v, "'صفحة الإدخالات للإيجارات'!A:ZZ", 1) for k, v in IDS.items()}
        return dfs

# --- 4. دوال مساعدة (HELPERS) ---
def get_col_by_letter(df, letter):
    def letter_to_index(col_str):
        num = 0
        for c in col_str:
            if c.upper() in string.ascii_uppercase: num = num * 26 + (ord(c.upper()) - ord('A')) + 1
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

def format_egp(x): return f"{x:,.0f} ج.م"
def format_usd(x): return f"${x:,.0f}"
def format_eur(x): return f"€{x:,.0f}"

def parse_ar_date(x):
    if pd.isna(x): return pd.NaT
    s = str(x).strip().replace("صباحًا", "AM").replace("مساءً", "PM").replace("ص", "AM").replace("م", "PM")
    try: return pd.to_datetime(s)
    except: return pd.NaT

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

# --- 5. MODULE 1: OPERATIONS ---
def show_operations(dfs):
    st.title("🏠 العمليات اليومية")
    if not dfs: return
    df_orders, df_cars = dfs['orders'], dfs['cars']

    with st.expander("🔎 فلاتر البحث", expanded=False):
        c1, c2 = st.columns(2)
        period_type = c1.selectbox("نوع الفترة", ["شهر", "ربع سنوي", "سنة"])
        sel_year = c2.selectbox("السنة", [2024, 2025, 2026, 2027], index=2)
        c3, c4 = st.columns(2)
        if period_type == "شهر": sel_spec = c3.selectbox("الشهر", range(1, 13), index=get_now().month-1)
        elif period_type == "ربع سنوي": sel_spec = c3.selectbox("الربع", [1, 2, 3, 4], index=0)
        else: sel_spec = 0 
        fleet_status = c4.selectbox("عرض الأسطول", ["السيارات النشطة", "الكل", "السيارات المتوقفة"], index=0)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)
    today = get_now()
    active_rentals = 0
    car_status_map = {} 
    
    col_start = get_col_by_letter(df_orders, 'L') 
    col_end = get_col_by_letter(df_orders, 'T')   
    col_car_ord = get_col_by_letter(df_orders, 'D') 
    
    if col_start and col_car_ord:
        for _, row in df_orders.iterrows():
            try:
                cid = clean_id_tag(row[col_car_ord])
                s = parse_ar_date(row[col_start])
                e = parse_ar_date(row[col_end])
                if pd.notnull(s) and pd.notnull(e) and s <= today <= e: car_status_map[cid] = "🔴" 
            except: continue

    car_map = {} 
    active_fleet_count = 0
    sunburst_data = []
    
    col_code, col_status, col_brand, col_model = get_col_by_letter(df_cars, 'A'), get_col_by_letter(df_cars, 'AZ'), get_col_by_letter(df_cars, 'B'), get_col_by_letter(df_cars, 'E')
    plate_cols = ['W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC']

    if col_code and col_status:
        valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
        if fleet_status == "السيارات النشطة": cars_subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        elif fleet_status == "السيارات المتوقفة": cars_subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        else: cars_subset = valid_rows

        active_fleet_count = len(cars_subset)
        for _, row in cars_subset.iterrows(): 
            try:
                c_id, c_name = clean_id_tag(row[col_code]), f"{row[col_brand]} {row[col_model]}"
                plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                indicator = car_status_map.get(c_id, "🟢") 
                if indicator == "🔴": active_rentals += 1 
                car_map[c_id] = f"{indicator} {c_name} | {plate.strip()}"
                sunburst_data.append({'Brand': str(row[col_brand]).strip(), 'Model': str(row[col_model]).strip(), 'Count': 1})
            except: continue

    returning_today, future_orders, timeline_data = 0, 0, []
    
    if col_start and col_end and col_car_ord:
        for _, row in df_orders.iterrows():
            try:
                s_date, e_date = parse_ar_date(row[col_start]), parse_ar_date(row[col_end])
                if pd.isna(s_date) or pd.isna(e_date) or not (s_date <= end_range and e_date >= start_range): continue
                car_id_clean = clean_id_tag(row[col_car_ord])
                if car_id_clean not in car_map: continue
                
                status = 'مكتمل'
                if s_date <= today <= e_date: status = 'نشط'
                elif s_date > today: status, future_orders = 'قادم', future_orders + 1
                if e_date.date() == today.date(): returning_today += 1
                
                timeline_data.append({'السيارة': car_map[car_id_clean], 'البدء': s_date, 'الانتهاء': e_date, 'الحالة': status})
            except: continue

    st.subheader("📊 ملخص الأسطول")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("إجمالي السيارات", active_fleet_count)
    k2.metric("إيجارات نشطة", active_rentals)
    k3.metric("متاح الآن", active_fleet_count - active_rentals)
    k4.metric("نسبة التشغيل", f"{(active_rentals / active_fleet_count * 100) if active_fleet_count > 0 else 0.0:.1f}%")
    
    c1, c2 = st.columns(2)
    with c1:
        if sunburst_data:
            fig = px.sunburst(pd.DataFrame(sunburst_data), path=['Brand', 'Model'], values='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=250, margin=dict(t=0, l=0, r=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(names=['مؤجر', 'متاح'], values=[active_rentals, active_fleet_count - active_rentals], hole=0.5, color_discrete_map={'مؤجر':'#ff4b4b', 'متاح':'#00C853'})
        fig.update_layout(height=250, margin=dict(t=0, l=0, r=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    df_timeline = pd.DataFrame(timeline_data) if timeline_data else pd.DataFrame(columns=['السيارة', 'البدء', 'الانتهاء', 'الحالة'])
    for car_name in sorted(list(car_map.values())):
        if car_name not in df_timeline['السيارة'].values:
            df_timeline = pd.concat([df_timeline, pd.DataFrame([{'السيارة': car_name, 'البدء': pd.NaT, 'الانتهاء': pd.NaT, 'الحالة': 'نشط'}])], ignore_index=True)

    if not df_timeline.empty:
        fig = px.timeline(df_timeline, x_start="البدء", x_end="الانتهاء", y="السيارة", color="الحالة", color_discrete_map={"نشط": "#ff4b4b", "قادم": "#9b59b6", "مكتمل": "#95a5a6"})
        fig.update_yaxes(autorange="reversed", categoryorder='array', categoryarray=sorted(list(car_map.values())), type='category')
        fig.update_layout(height=max(300, len(car_map) * 35), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white", size=10), margin=dict(l=10, r=10, t=10, b=10))
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
        st.plotly_chart(fig, use_container_width=True)

# --- 6. MODULE 2: VEHICLE 360 ---
def show_vehicle_360(dfs):
    st.title("🚗 ملف السيارات")
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
                        c_id, c_label = clean_id_tag(row[col_code]), f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                        plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                        car_options[f"[{row[col_code]}] {c_label} | {plate.strip()}"] = c_id
                    except: continue
            select_all = st.checkbox("تحديد الكل")
            selected_labels = st.multiselect("المركبات", list(car_options.keys()), default=list(car_options.keys()) if select_all else [])
            selected_ids = [car_options[l] for l in selected_labels]

        st.markdown("---")
        tf1, tf2, tf3, tf4 = st.columns(4)
        period_type = tf1.selectbox("عرض", ["شهر", "ربع سنوي", "سنة"], key='v360_p')
        sel_year = tf2.selectbox("السنة", [2024, 2025, 2026], index=2, key='v360_y')
        sel_spec = tf3.selectbox("الشهر/الربع", range(1, 13) if period_type == "شهر" else ([1, 2, 3, 4] if period_type == "ربع سنوي" else [0]), index=get_now().month-1 if period_type == "شهر" else 0)
        show_active = tf4.checkbox("إخفاء الفارغ", value=False)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)
    if not selected_ids: st.info("👈 اختر المركبات."); return

    trips_data, maint_list, exp_list = [], [], []
    total_revenue, total_maint, total_exp = 0.0, 0.0, 0.0
    
    col_ord_start, col_ord_end, col_ord_cost, col_ord_car, col_ord_id = get_col_by_letter(df_orders, 'L'), get_col_by_letter(df_orders, 'T'), get_col_by_letter(df_orders, 'AU'), get_col_by_letter(df_orders, 'D'), get_col_by_letter(df_orders, 'A')
    col_ord_loc_start, col_ord_loc_end, col_ord_dur_txt = get_col_by_letter(df_orders, 'M'), get_col_by_letter(df_orders, 'U'), get_col_by_letter(df_orders, 'V')

    if col_ord_start:
        for _, row in df_orders.iterrows():
            cid = clean_id_tag(row[col_ord_car])
            if cid in selected_ids:
                d_s, d_e = parse_ar_date(row[col_ord_start]), parse_ar_date(row[col_ord_end])
                if pd.notnull(d_s) and start_range <= d_s <= end_range:
                    rev = clean_currency(row[col_ord_cost])
                    total_revenue += rev
                    
                    dur_txt = str(row[col_ord_dur_txt]) if pd.notnull(row[col_ord_dur_txt]) else ""
                    days_calc = (d_e - d_s).days if pd.notnull(d_e) else 1
                    if days_calc == 0: days_calc = 1
                    
                    trips_data.append({
                        "السيارة": [k for k, v in car_options.items() if v == cid][0],
                        "رقم الطلب": row[col_ord_id],
                        "البدء": f"{d_s.strftime('%Y-%m-%d %I:%M %p')} - {row[col_ord_loc_start]}",
                        "الانتهاء": f"{d_e.strftime('%Y-%m-%d %I:%M %p')} - {row[col_ord_loc_end]}" if pd.notnull(d_e) else "-",
                        "المدة": dur_txt,
                        "اليومية": format_egp(rev / days_calc),
                        "الإجمالي": format_egp(rev)
                    })

    col_exp_car, col_exp_amt, col_exp_y, col_exp_m, col_exp_d, col_exp_rec = get_col_by_letter(df_car_exp, 'S'), get_col_by_letter(df_car_exp, 'Z'), get_col_by_letter(df_car_exp, 'Y'), get_col_by_letter(df_car_exp, 'X'), get_col_by_letter(df_car_exp, 'W'), get_col_by_letter(df_car_exp, 'A')
    col_exp_type_ar, col_exp_maint_ar, col_exp_stmt_ar = get_col_by_letter(df_car_exp, 'E'), get_col_by_letter(df_car_exp, 'H'), get_col_by_letter(df_car_exp, 'K') 
    col_ref_q, col_ref_r, col_ref_t, col_ref_s = get_col_by_letter(df_car_exp, 'Q'), get_col_by_letter(df_car_exp, 'R'), get_col_by_letter(df_car_exp, 'T'), get_col_by_letter(df_car_exp, 'S') 

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            cid = clean_id_tag(row[col_exp_car])
            if cid in selected_ids:
                try:
                    y, m, d_val = int(clean_currency(row[col_exp_y])), int(clean_currency(row[col_exp_m])), int(clean_currency(row[col_exp_d]))
                    valid = (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec])
                    if valid:
                        amt = clean_currency(row[col_exp_amt])
                        type_str = str(row[col_exp_type_ar]).strip() 
                        is_maint = False
                        if "صيانات" in type_str: is_maint, display_name = True, str(row[col_exp_maint_ar])
                        elif "تعاقد" in type_str: display_name = f"{type_str} / {row[col_ref_q]} - {row[col_ref_r]}"
                        elif "رد تامين" in type_str: display_name = f"{type_str} / {row[col_ref_t]}"
                        elif "عمولة" in type_str: display_name = f"{type_str} / {row[col_ref_s]}"
                        elif "نثريات حركة" in type_str: display_name = f"{type_str} / {row[col_exp_stmt_ar]}"
                        elif "تشغيل" in type_str: display_name = f"{type_str} / {row[col_exp_stmt_ar]} - {row[col_ref_t]}"
                        else: display_name = f"{type_str} - {str(row[col_exp_stmt_ar])}"

                        entry = {"رقم السجل": row[col_exp_rec], "السيارة": [k for k, v in car_options.items() if v == cid][0], "التاريخ": f"{y}-{m:02d}-{d_val:02d}", "البند": display_name, "التكلفة": format_egp(amt)}
                        if is_maint: maint_list.append(entry); total_maint += amt
                        else: exp_list.append(entry); total_exp += amt
                except: continue

    if show_active and not trips_data: st.warning("لا يوجد بيانات."); return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("الإيراد الكلي", format_egp(total_revenue))
    k2.metric("الصيانة", format_egp(total_maint), delta_color="inverse")
    k3.metric("المصروفات", format_egp(total_exp), delta_color="inverse")
    k4.metric("الصافي", format_egp(total_revenue - total_maint - total_exp))
    
    t1, t2, t3 = st.tabs(["الرحلات", "الصيانة", "المصروفات"])
    with t1: st.dataframe(pd.DataFrame(trips_data), use_container_width=True) if trips_data else st.info("فارغ")
    with t2: st.dataframe(pd.DataFrame(maint_list), use_container_width=True) if maint_list else st.info("فارغ")
    with t3: st.dataframe(pd.DataFrame(exp_list), use_container_width=True) if exp_list else st.info("فارغ")

# --- 7. MODULE 3: CRM ---
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
                car_display_map[clean_id_tag(row[col_code])] = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]} | " + "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])]).strip()
            except: continue

    client_id_map, client_db = {}, {}
    col_cl_id, col_cl_first, col_cl_last = get_col_by_letter(df_clients, 'A'), get_col_by_letter(df_clients, 'C'), get_col_by_letter(df_clients, 'D')
    if col_cl_id:
        for _, row in df_clients.iterrows():
            try:
                cid, full_name = clean_client_code(row[col_cl_id]), f"{str(row[col_cl_first]) if pd.notnull(row[col_cl_first]) else ''} {str(row[col_cl_last]) if pd.notnull(row[col_cl_last]) else ''}".strip()
                if not full_name: continue
                client_id_map[cid] = full_name
                client_db[full_name] = {'Display': f"[{cid}] {full_name}", 'Name': full_name, 'Spend': 0, 'Trips': 0, 'History': [], 'DepositHeld': 0, 'PaidUSD': 0, 'PaidEUR': 0}
            except: continue

    col_ord_name, col_ord_cost, col_ord_s, col_ord_e, col_ord_car, col_ord_id = get_col_by_letter(df_orders, 'B'), get_col_by_letter(df_orders, 'AU'), get_col_by_letter(df_orders, 'L'), get_col_by_letter(df_orders, 'T'), get_col_by_letter(df_orders, 'D'), get_col_by_letter(df_orders, 'A')
    col_ord_dep_held, col_ord_usd, col_ord_eur, col_ord_dur_txt = get_col_by_letter(df_orders, 'AW'), get_col_by_letter(df_orders, 'AY'), get_col_by_letter(df_orders, 'AZ'), get_col_by_letter(df_orders, 'V')

    if col_ord_name:
        for _, row in df_orders.iterrows():
            try:
                raw_val = clean_client_code(row[col_ord_name])
                if not raw_val or raw_val == "nan": continue
                real_name = client_id_map.get(raw_val, raw_val) 
                if real_name not in client_db: client_db[real_name] = {'Display': f"[?] {real_name}", 'Name': real_name, 'Spend': 0, 'Trips': 0, 'History': [], 'DepositHeld': 0, 'PaidUSD': 0, 'PaidEUR': 0}
                
                rec = client_db[real_name]
                amt, usd, eur, dep = clean_currency(row[col_ord_cost]), clean_currency(row[col_ord_usd]), clean_currency(row[col_ord_eur]), clean_currency(row[col_ord_dep_held])
                s, e, cid = parse_ar_date(row[col_ord_s]), parse_ar_date(row[col_ord_e]), clean_id_tag(row[col_ord_car])
                dur_txt = str(row[col_ord_dur_txt]) if pd.notnull(row[col_ord_dur_txt]) else ""
                
                status = "نشط" if pd.notnull(s) and pd.notnull(e) and s <= get_now() <= e else ("قادم" if pd.notnull(s) and s > get_now() else "مكتمل")
                
                rec['Spend'] += amt
                rec['PaidUSD'] += usd
                rec['PaidEUR'] += eur
                rec['DepositHeld'] += dep
                rec['Trips'] += 1
                rec['History'].append({
                    "رقم الطلب": row[col_ord_id], "السيارة": car_display_map.get(cid, cid),
                    "البدء": s.strftime("%Y-%m-%d") if pd.notnull(s) else "-", "الانتهاء": e.strftime("%Y-%m-%d") if pd.notnull(e) else "-",
                    "المدة": dur_txt, "التكلفة": format_egp(amt), "الحالة": status, "وديعة معلقة": format_egp(dep)
                })
            except: continue

    search = st.text_input("🔍 بحث عن عميل", "")
    df_crm = pd.DataFrame([{'Display': v['Display'], 'الإنفاق EGP': format_egp(v['Spend']), 'الإنفاق USD': format_usd(v['PaidUSD']), 'الإنفاق EUR': format_eur(v['PaidEUR']), 'ودائع معلقة': format_egp(v['DepositHeld']), 'رحلات': v['Trips'], 'Key': v['Name'], 'SpendRaw': v['Spend']} for v in client_db.values()])
    
    if not df_crm.empty:
        df_crm = df_crm.sort_values('SpendRaw', ascending=False)
        if search: df_crm = df_crm[df_crm['Display'].str.contains(search, case=False, na=False)]

        c1, c2, c3 = st.columns(3)
        c1.metric("إجمالي العملاء", len(client_db))
        c2.metric("الأكثر إنفاقاً", df_crm.iloc[0]['Display'].split("] ")[-1] if len(df_crm)>0 else "-")
        c3.metric("إجمالي الودائع المعلقة", format_egp(sum(v['DepositHeld'] for v in client_db.values())))

        st.divider()
        col_list, col_detail = st.columns([1, 2])
        with col_list:
            selection = st.dataframe(df_crm[['Display', 'الإنفاق EGP', 'رحلات', 'ودائع معلقة']], use_container_width=True, height=500, on_select="rerun", selection_mode="single-row", hide_index=True)
        with col_detail:
            if selection.selection.rows:
                client_data = client_db[df_crm.iloc[selection.selection.rows[0]]['Key']]
                st.info(f"**{client_data['Display']}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("إجمالي (EGP)", format_egp(client_data['Spend']))
                m2.metric("إجمالي (USD)", format_usd(client_data['PaidUSD']))
                m3.metric("إجمالي (EUR)", format_eur(client_data['PaidEUR']))
                m4.metric("وديعة مطلوبة للرد", format_egp(client_data['DepositHeld']), delta_color="inverse" if client_data['DepositHeld']>0 else "off")
                if client_data['History']: st.dataframe(pd.DataFrame(client_data['History']), use_container_width=True, hide_index=True)
            else: st.info("👈 اختر عميلاً.")

# --- 8. MODULE 4: FINANCIAL HQ (MASTER AUDIT) ---
def show_financial_hq(dfs):
    st.title("💰 الإدارة المالية")
    if not dfs: return

    df_coll, df_exp, df_car_exp, df_cars, df_orders = dfs['collections'], dfs['expenses'], dfs['car_expenses'], dfs['cars'], dfs['orders']

    with st.expander("🗓️ إعدادات الفترة", expanded=True):
        f1, f2, f3 = st.columns(3)
        period_type = f1.selectbox("نوع العرض", ["شهر", "ربع سنوي", "سنة"], key='fin_p')
        sel_year = f2.selectbox("السنة المالية", [2024, 2025, 2026, 2027], index=2, key='fin_y')
        calc_method = f3.selectbox("طريقة الحساب", ["عن الفترة المحددة", "تراكمي حتى الآن"])
        f4, f5 = st.columns(2)
        if period_type == "شهر": sel_spec = f4.selectbox("الشهر", range(1, 13), index=get_now().month-1, key='fin_m')
        elif period_type == "ربع سنوي": sel_spec = f4.selectbox("الربع", [1, 2, 3, 4], index=0, key='fin_q')
        else: sel_spec = 0

    start_date, end_date = get_date_filter_range(period_type, sel_year, sel_spec)
    
    inflow_cats, expense_cats = {}, {}
    cash_in, cash_out = 0.0, 0.0
    
    # 1. Orders Data (For Deposits in Audit Tab)
    deposits_collected, deposits_refunded = 0.0, 0.0
    col_ord_s, col_ord_dep_coll, col_ord_dep_ref = get_col_by_letter(df_orders, 'L'), get_col_by_letter(df_orders, 'AB'), get_col_by_letter(df_orders, 'AV')
    if col_ord_s:
        for _, row in df_orders.iterrows():
            try:
                s = parse_ar_date(row[col_ord_s])
                if pd.notnull(s) and start_date <= s <= end_date:
                    deposits_collected += clean_currency(row[col_ord_dep_coll])
                    deposits_refunded += clean_currency(row[col_ord_dep_ref])
            except: continue

    # 2. Collections
    col_coll_amt, col_coll_y, col_coll_m = get_col_by_letter(df_coll, 'R'), get_col_by_letter(df_coll, 'Q'), get_col_by_letter(df_coll, 'P')
    if col_coll_amt:
        for _, row in df_coll.iterrows():
            try:
                y, m = int(clean_currency(row[col_coll_y])), int(clean_currency(row[col_coll_m]))
                if (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]):
                    amt = clean_currency(row[col_coll_amt])
                    cash_in += amt
                    inflow_cats["تأجير"] = inflow_cats.get("تأجير", 0) + amt
            except: continue

    # 3. Expenses
    col_exp_amt, col_exp_y, col_exp_m, col_exp_type = get_col_by_letter(df_exp, 'X'), get_col_by_letter(df_exp, 'W'), get_col_by_letter(df_exp, 'V'), get_col_by_letter(df_exp, 'I')
    if col_exp_amt:
        for _, row in df_exp.iterrows():
            try:
                y, m = int(clean_currency(row[col_exp_y])), int(clean_currency(row[col_exp_m]))
                if (period_type=="سنة" and y==sel_year) or (period_type=="شهر" and y==sel_year and m==sel_spec) or (period_type=="ربع سنوي" and y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]):
                    amt = clean_currency(row[col_exp_amt])
                    cash_out += amt
                    cat = str(row[col_exp_type]).strip() if pd.notnull(row[col_exp_type]) else "نثريات"
                    expense_cats[cat] = expense_cats.get(cat, 0) + amt
            except: continue

    # 4. Car Expenses (Ledger & Audit)
    col_cexp_amt, col_cexp_y, col_cexp_m, col_cexp_car, col_cexp_id_g = get_col_by_letter(df_car_exp, 'Z'), get_col_by_letter(df_car_exp, 'Y'), get_col_by_letter(df_car_exp, 'X'), get_col_by_letter(df_car_exp, 'S'), get_col_by_letter(df_car_exp, 'G') 
    ledger_history, contracting_audit = [], {}
    
    if col_cexp_amt:
        for _, row in df_car_exp.iterrows():
            try:
                amt, cid, type_id = clean_currency(row[col_cexp_amt]), clean_id_tag(row[col_cexp_car]), str(row[col_cexp_id_g]).strip()
                y, m = int(clean_currency(row[col_cexp_y])), int(clean_currency(row[col_cexp_m]))
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

    for _, car in df_cars.iterrows():
        try:
            if not any(x in str(car[col_status]).lower() for x in ['valid', 'active', 'ساري']): continue
            cid = clean_id_tag(car[col_code])
            owner_name = f"{car[col_owner_f]} {car[col_owner_l]}".strip()
            cname = f"{car[col_car_name]} {car[get_col_by_letter(df_cars, 'E')]}"
            plate = "".join([str(car[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(car[get_col_by_letter(df_cars, p)])]).strip()
            
            search_key = f"[{cid}] {owner_name} - {cname} ({str(car[col_model_yr])}) | {plate}"
            cid_to_meta[cid] = {'Label': search_key, 'Owner': owner_name, 'Car': cname}
            
            s_date = pd.to_datetime(car[col_contract_start], errors='coerce')
            if pd.isna(s_date): continue
            
            base_fee, freq_days, deduct_pct, brokerage = clean_currency(car[col_monthly_fee]), clean_currency(car[col_pay_freq]), clean_currency(car[col_deduct_pct]), clean_currency(car[col_brokerage])
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

    tab1, tab2, tab3, tab4 = st.tabs(["التدفق النقدي", "الأرباح والخسائر", "كشف حساب الملاك", "التفاصيل والمراجعة"])
    
    with tab1:
        cat_cols = st.columns(4)
        cat_cols[0].metric("دفعات تعاقد", format_egp(expense_cats.get("دفعات تعاقد", 0)))
        cat_cols[1].metric("صيانة / مخالفات", format_egp(expense_cats.get("صيانة / مخالفات", 0)))
        cat_cols[2].metric("نثريات / تشغيل", format_egp(sum(v for k,v in expense_cats.items() if k not in ["دفعات تعاقد", "صيانة / مخالفات"])))
        cat_cols[3].metric("عمولات", format_egp(expense_cats.get("عمولات", 0)))
        
        st.divider()
        net = cash_in - cash_out
        c1, c2, c3 = st.columns(3)
        c1.metric("وارد", format_egp(cash_in))
        c2.metric("صادر", format_egp(cash_out), delta_color="inverse")
        c3.metric("الصافي", format_egp(net), delta_color="normal" if net>=0 else "inverse")
        
        chart1, chart2 = st.columns(2)
        with chart1:
            if inflow_cats:
                fig_in = px.pie(pd.DataFrame(list(inflow_cats.items()), columns=['Source', 'Amount']), values='Amount', names='Source', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
                fig_in.update_layout(height=300, margin=dict(t=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
                st.plotly_chart(fig_in, use_container_width=True)
        with chart2:
            if expense_cats:
                fig_out = px.bar(pd.DataFrame(list(expense_cats.items()), columns=['Category', 'Amount']), x='Category', y='Amount', color='Category')
                fig_out.update_layout(height=300, margin=dict(t=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", showlegend=False)
                st.plotly_chart(fig_out, use_container_width=True)

    with tab2:
        items = [('الإيرادات', cash_in)]
        for k, v in expense_cats.items(): items.append((k, -v))
        df_waterfall = pd.DataFrame(items, columns=['Category', 'Amount'])
        net_profit = df_waterfall['Amount'].sum()
        
        k1, k2 = st.columns(2)
        k1.metric("الإيرادات", format_egp(cash_in))
        k2.metric("صافي الربح", format_egp(net_profit), delta_color="normal" if net_profit>=0 else "inverse")
        
        fig = go.Figure(go.Waterfall(
            name = "P&L", orientation = "v", measure = ["relative"] * len(df_waterfall) + ["total"],
            x = df_waterfall['Category'].tolist() + ["الصافي"], y = df_waterfall['Amount'].tolist() + [0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}}, decreasing = {"marker":{"color":"#ef5350"}}, increasing = {"marker":{"color":"#66bb6a"}}, totals = {"marker":{"color":"#42a5f5"}}
        ))
        fig.update_layout(height=400, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df_all = pd.DataFrame(ledger_history)
        if not df_all.empty:
            df_all['Search_Label'] = df_all['CID'].map(lambda x: cid_to_meta.get(x, {}).get('Label', 'Unknown'))
            if calc_method == "عن الفترة المحددة": df_filtered = df_all[(df_all['Date'] >= start_date) & (df_all['Date'] <= end_date)].copy()
            else: df_filtered = df_all[df_all['Date'] <= get_now()].copy() 

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
                m2.metric("إجمالي المدفوعات/الخصم", format_egp(abs(total_paid)))
                m3.metric("الرصيد الصافي", f"{'⬇️🔴' if net_bal > 0 else '⬆️🟢'} {format_egp(abs(net_bal))}")
                
                st.divider()
                display = final_data[['Date', 'Search_Label', 'Icon', 'Type', 'Amount']].copy()
                display.columns = ['التاريخ', 'المركبة/المالك', ' ', 'البيان', 'القيمة']
                display['القيمة'] = display['القيمة'].apply(format_egp)
                display['التاريخ'] = display['التاريخ'].dt.strftime('%Y-%m-%d')
                st.dataframe(display, use_container_width=True)
            else: st.warning("يرجى اختيار مالك واحد على الأقل.")
        else: st.info("لا توجد بيانات مالية.")

    with tab4:
        st.subheader(f"📋 التفاصيل والمراجعة ({period_type})")
        
        audit_data = []
        for cid, data in contracting_audit.items():
            diff = data['Paid'] - data['Due'] 
            status = "متوازن"
            if diff > 100: status = "مدفوع بالزيادة 🟢"
            elif diff < -100: status = "معلق (عجز) 🔴"
            audit_data.append({
                "السيارة": cid_to_meta.get(cid, {}).get('Car', cid),
                "المستحق": format_egp(data['Due']), "المدفوع": format_egp(data['Paid']),
                "الفارق": format_egp(diff), "الحالة": status
            })
        
        if audit_data:
            st.markdown("##### مراجعة التعاقدات للملاك")
            st.dataframe(pd.DataFrame(audit_data), use_container_width=True)
        else: st.info("لا توجد تعاقدات مستحقة في هذه الفترة.")
            
        st.divider()
        st.markdown("##### حركة الودائع والتأمين من الطلبات")
        d1, d2, d3 = st.columns(3)
        d1.metric("تأمين محصل", format_egp(deposits_collected)) 
        d2.metric("تأمين مسترد", format_egp(deposits_refunded))
        d3.metric("صافي المحتجز للعملاء", format_egp(deposits_collected - deposits_refunded), delta_color="inverse")

# --- 9. MODULE 5: RISK RADAR ---
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

    for _, row in df_cars.iterrows():
        try:
            if not any(x in str(row[col_status]).lower() for x in ['valid', 'active', 'ساري']): continue
            cid, cname = clean_id_tag(row[col_code]), f"[{clean_id_tag(row[col_code])}] {row[col_name]} {row[col_model]}"
            plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])]).strip()
            
            if col_lic_status and any(x in str(row[col_lic_status]).lower() for x in ['valid', 'active', 'ساري']):
                d_lic, d_exam = pd.to_datetime(row[col_lic_end], errors='coerce'), pd.to_datetime(row[col_exam_end], errors='coerce')
                target, reason = None, "ترخيص"
                if d_lic and d_exam: target, reason = (d_lic, "ترخيص + فحص") if d_lic == d_exam else ((d_lic, "ترخيص") if d_lic < d_exam else (d_exam, "فحص"))
                elif d_lic: target, reason = d_lic, "ترخيص"
                elif d_exam: target, reason = d_exam, "فحص"
                if target:
                    days = (target - today).days
                    bucket = "خطر مرتفع (0-3 أشهر)" if days <= 90 else ("خطر متوسط (3-6 أشهر)" if days <= 180 else "خطر منخفض (> 6 أشهر)")
                    risks['License'].append({'السيارة': cname, 'اللوحة': plate, 'السبب': reason, 'الاستحقاق': target.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

            if col_ins_status and ("yes" in str(row[col_ins_status]).lower() or "يوجد" in str(row[col_ins_status]).lower()):
                d = pd.to_datetime(row[col_ins_end], errors='coerce')
                if pd.notnull(d):
                    days = (d - today).days
                    bucket = "خطر مرتفع (0-3 أشهر)" if days <= 90 else ("خطر متوسط (3-6 أشهر)" if days <= 180 else "خطر منخفض (> 6 أشهر)")
                    risks['Insurance'].append({'السيارة': cname, 'اللوحة': plate, 'الاستحقاق': d.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

            if col_con_end:
                d = pd.to_datetime(row[col_con_end], errors='coerce')
                if pd.notnull(d):
                    days = (d - today).days
                    bucket = "خطر مرتفع (0-3 أشهر)" if days <= 90 else ("خطر متوسط (3-6 أشهر)" if days <= 180 else "خطر منخفض (> 6 أشهر)")
                    risks['Contract'].append({'السيارة': cname, 'اللوحة': plate, 'الاستحقاق': d.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})
        except: continue

    t1, t2, t3 = st.tabs(["📄 الترخيص", "🛡️ التأمين", "📝 العقود"])
    def render_tab(category):
        if not risks[category]: st.success("✅ الكل سليم."); return
        df = pd.DataFrame(risks[category]).sort_values('Days')
        for title, label in [("🔴 خطر مرتفع (0-3 أشهر)", "خطر مرتفع (0-3 أشهر)"), ("🟡 خطر متوسط (3-6 أشهر)", "خطر متوسط (3-6 أشهر)"), ("🟢 خطر منخفض (> 6 أشهر)", "خطر منخفض (> 6 أشهر)")]:
            subset = df[df['التصنيف'] == label]
            with st.expander(f"{title} [{len(subset)}]", expanded=(label == "خطر مرتفع (0-3 أشهر)")):
                if not subset.empty: st.dataframe(subset.drop(columns=['التصنيف', 'Days']), use_container_width=True)
                else: st.info("لا يوجد")

    with t1: render_tab('License')
    with t2: render_tab('Insurance')
    with t3: render_tab('Contract')

# --- 10. القائمة الجانبية (NAV) ---
st.sidebar.title("🚘 نظام التأجير")
page = st.sidebar.radio("", ["العمليات", "ملف السيارات", "إدارة العملاء", "الإدارة المالية", "رادار المخاطر"])
st.sidebar.markdown("---")
dfs = load_data_v3()
if page == "العمليات": show_operations(dfs)
elif page == "ملف السيارات": show_vehicle_360(dfs)
elif page == "إدارة العملاء": show_crm(dfs)
elif page == "الإدارة المالية": show_financial_hq(dfs)
elif page == "رادار المخاطر": show_risk_radar(dfs)
