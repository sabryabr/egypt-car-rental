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
st.set_page_config(page_title="Egypt Rental OS 3.0", layout="wide", page_icon="🚘", initial_sidebar_state="auto")

# --- 2. ENHANCED CSS (CENTERED TABLES & RTL) ---
st.markdown("""
<style>
    /* Global RTL & Font */
    .main { direction: rtl; font-family: 'Tajawal', sans-serif; background-color: #0e1117; color: white; text-align: right; }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #262730; border: 1px solid #464b5d; border-radius: 8px; padding: 10px; 
        color: white; height: auto; min-height: 80px; overflow: hidden; text-align: right;
    }
    
    /* Tables: Center Align & Autofit */
    .stDataFrame { direction: rtl; width: 100%; }
    .stDataFrame div[data-testid="stHorizontalBlock"] { width: 100%; }
    th { text-align: center !important; }
    td { text-align: center !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; margin-bottom: 1rem; flex-wrap: wrap; direction: rtl; }
    .stTabs [data-baseweb="tab"] { height: 40px; padding: 0 15px; font-size: 0.9rem; flex-grow: 1; }
    
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
        st.error("⚠️ خطأ: يرجى إضافة بيانات الاعتماد (Secrets).")
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
            st.warning(f"⚠️ خطأ في التحميل ({range_name}): {str(e)}")
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
    # Full number format: 8,800
    return f"{x:,.0f} ج.م"

def get_date_filter_range(period_type, year, specifier):
    if period_type == "سنة":
        return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)
    elif period_type == "ربع سنوي":
        q_map = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
        s_m, e_m = q_map[specifier]
        _, last_day = calendar.monthrange(year, e_m)
        return datetime(year, s_m, 1), datetime(year, e_m, last_day, 23, 59, 59)
    else: 
        _, last_day = calendar.monthrange(year, specifier)
        return datetime(year, specifier, 1), datetime(year, specifier, last_day, 23, 59, 59)

# --- 5. MODULE 1: OPERATIONS (LIVE INDICATORS) ---
def show_operations(dfs):
    st.title("🏠 مركز العمليات")
    if not dfs: return

    df_orders = dfs['orders']
    df_cars = dfs['cars']

    with st.expander("🔎 أدوات الفلترة", expanded=False):
        c1, c2 = st.columns(2)
        period_type = c1.selectbox("نوع الفترة", ["شهر", "ربع سنوي", "سنة"])
        sel_year = c2.selectbox("السنة", [2024, 2025, 2026, 2027], index=2)
        c3, c4 = st.columns(2)
        if period_type == "شهر":
            sel_spec = c3.selectbox("الشهر", range(1, 13), index=datetime.now().month-1)
        elif period_type == "ربع سنوي":
            sel_spec = c3.selectbox("الربع", [1, 2, 3, 4], index=0)
        else: sel_spec = 0 
        fleet_status = c4.selectbox("حالة الأسطول", ["السيارات النشطة", "الكل", "السيارات المتوقفة"], index=0)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)

    today = datetime.now()
    active_rentals = 0
    car_status_map = {} 
    
    col_start = get_col_by_letter(df_orders, 'L')
    col_end = get_col_by_letter(df_orders, 'V')
    col_car_ord = get_col_by_letter(df_orders, 'C')
    col_client = get_col_by_letter(df_orders, 'B')
    
    if col_start and col_car_ord:
        for _, row in df_orders.iterrows():
            try:
                cid = clean_id_tag(row[col_car_ord])
                s = pd.to_datetime(row[col_start], errors='coerce')
                e = pd.to_datetime(row[col_end], errors='coerce')
                if pd.notnull(s) and pd.notnull(e):
                    if s <= today <= e:
                        car_status_map[cid] = "🔴" 
            except: continue

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
        if fleet_status == "السيارات النشطة":
            cars_subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        elif fleet_status == "السيارات المتوقفة":
            cars_subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
        else: cars_subset = valid_rows

        active_fleet_count = len(cars_subset)
        for _, row in cars_subset.iterrows(): 
            try:
                c_id = clean_id_tag(row[col_code])
                c_name = f"{row[col_brand]} {row[col_model]}"
                plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                
                indicator = car_status_map.get(c_id, "🟢") 
                if indicator == "🔴": active_rentals += 1 
                
                car_map[c_id] = f"{indicator} {c_name} | {plate.strip()}"
                sunburst_data.append({'Brand': str(row[col_brand]).strip(), 'Model': str(row[col_model]).strip(), 'Count': 1})
            except: continue

    returning_today = 0
    future_orders = 0
    timeline_data = []
    
    if col_start and col_end and col_car_ord:
        for _, row in df_orders.iterrows():
            try:
                s_date = pd.to_datetime(row[col_start], errors='coerce')
                e_date = pd.to_datetime(row[col_end], errors='coerce')
                if pd.isna(s_date) or pd.isna(e_date): continue
                if not (s_date <= end_range and e_date >= start_range): continue

                car_id_clean = clean_id_tag(row[col_car_ord])
                if car_id_clean not in car_map: continue
                
                status = 'مكتمل'
                if s_date <= today <= e_date: 
                    status = 'نشط'
                elif s_date > today: 
                    status = 'قادم'
                    future_orders += 1
                if e_date.date() == today.date(): returning_today += 1
                
                timeline_data.append({
                    'Car': car_map[car_id_clean], 'Start': s_date, 'End': e_date,
                    'Client': str(row[col_client]) if col_client else "N/A", 'Status': status
                })
            except: continue

    available_cars = active_fleet_count - active_rentals
    utilization = (active_rentals / active_fleet_count * 100) if active_fleet_count > 0 else 0.0
    
    st.subheader("📊 ملخص الأسطول")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚗 الأسطول", active_fleet_count)
    k2.metric("⚡ إيجارات حية", active_rentals)
    k3.metric("🟢 متاح", available_cars)
    k4.metric("📈 نسبة التشغيل", f"{utilization:.1f}%")
    
    c1, c2 = st.columns(2)
    with c1:
        if sunburst_data:
            fig = px.sunburst(pd.DataFrame(sunburst_data), path=['Brand', 'Model'], values='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=250, margin=dict(t=0, l=0, r=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(names=['مؤجر', 'متاح'], values=[active_rentals, available_cars], hole=0.5, color_discrete_map={'مؤجر':'#ff4b4b', 'متاح':'#00C853'})
        fig.update_layout(height=250, margin=dict(t=0, l=0, r=0, b=0), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(f"**الجدول الزمني ({period_type})**")
    
    all_car_names = sorted(list(car_map.values()))
    df_timeline = pd.DataFrame(timeline_data) if timeline_data else pd.DataFrame(columns=['Car', 'Start', 'End', 'Status', 'Client'])

    for car_name in all_car_names:
        if car_name not in df_timeline['Car'].values:
            new_row = pd.DataFrame([{'Car': car_name, 'Start': pd.NaT, 'End': pd.NaT, 'Status': 'نشط', 'Client': ''}])
            df_timeline = pd.concat([df_timeline, new_row], ignore_index=True)

    if not df_timeline.empty:
        color_map = {"نشط": "#ff4b4b", "قادم": "#9b59b6", "مكتمل": "#95a5a6"}
        fig = px.timeline(df_timeline, x_start="Start", x_end="End", y="Car", color="Status", color_discrete_map=color_map, hover_data=["Client"])
        fig.update_yaxes(autorange="reversed", categoryorder='array', categoryarray=all_car_names, type='category')
        fig.update_layout(height=max(300, len(all_car_names) * 35), plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", 
                          font=dict(color="white", size=10), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(showgrid=True, gridcolor="#333", range=[start_range, end_range]))
        fig.add_vline(x=today.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#FF3D00")
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("لا يوجد سيارات.")

# --- 6. MODULE 2: VEHICLE 360 (FORMAT & DETAIL) ---
def show_vehicle_360(dfs):
    st.title("🚗 ملف السيارات")
    if not dfs: return

    df_cars = dfs['cars']
    df_orders = dfs['orders']
    df_car_exp = dfs['car_expenses']

    with st.expander("🔎 التحكم", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1: fleet_cat = st.radio("التصنيف", ["النشطة", "الأرشيف", "الكل"], horizontal=True)
        with col2:
            car_options = {}
            col_code = get_col_by_letter(df_cars, 'A')
            col_status = get_col_by_letter(df_cars, 'AZ')
            plate_cols = ['AC','AB','AA','Z','Y','X','W']
            if col_code and col_status:
                valid_rows = df_cars[df_cars[col_code].notna() & (df_cars[col_code].astype(str).str.strip() != "")]
                if fleet_cat == "النشطة": subset = valid_rows[valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                elif fleet_cat == "الأرشيف": subset = valid_rows[~valid_rows[col_status].astype(str).str.contains('Valid|Active|ساري', case=False, na=False)]
                else: subset = valid_rows 
                for _, row in subset.iterrows():
                    try:
                        c_id = clean_id_tag(row[col_code])
                        c_label = f"{row[get_col_by_letter(df_cars, 'B')]} {row[get_col_by_letter(df_cars, 'E')]}"
                        plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])])
                        car_options[f"[{row[col_code]}] {c_label} | {plate.strip()}"] = c_id
                    except: continue
            select_all = st.checkbox("تحديد الكل")
            default_sel = list(car_options.keys()) if select_all else []
            selected_labels = st.multiselect("المركبات", list(car_options.keys()), default=default_sel)
            selected_ids = [car_options[l] for l in selected_labels]

        st.markdown("---")
        tf1, tf2 = st.columns(2)
        period_type = tf1.selectbox("عرض", ["شهر", "ربع سنوي", "سنة"], key='v360_p')
        sel_year = tf2.selectbox("السنة", [2024, 2025, 2026], index=2, key='v360_y')
        tf3, tf4 = st.columns(2)
        if period_type == "شهر": sel_spec = tf3.selectbox("الشهر", range(1, 13), index=datetime.now().month-1, key='v360_m')
        elif period_type == "ربع سنوي": sel_spec = tf3.selectbox("الربع", [1, 2, 3, 4], index=0, key='v360_q')
        else: sel_spec = 0
        show_active = tf4.checkbox("إخفاء الفارغ", value=False)

    start_range, end_range = get_date_filter_range(period_type, sel_year, sel_spec)
    if not selected_ids: st.info("👈 اختر المركبات."); return

    trips_data, maint_list, exp_list = [], [], []
    total_revenue, total_maint, total_exp = 0.0, 0.0, 0.0
    
    col_ord_start = get_col_by_letter(df_orders, 'L') # Date Time
    col_ord_end = get_col_by_letter(df_orders, 'V') # Date Time
    col_ord_cost = get_col_by_letter(df_orders, 'AE')
    col_ord_car = get_col_by_letter(df_orders, 'C')
    col_ord_id = get_col_by_letter(df_orders, 'A')
    
    col_ord_loc_start = get_col_by_letter(df_orders, 'M') # Pickup
    col_ord_loc_end = get_col_by_letter(df_orders, 'W') # Dropoff

    if col_ord_start:
        for _, row in df_orders.iterrows():
            cid = clean_id_tag(row[col_ord_car])
            if cid in selected_ids:
                d_s = pd.to_datetime(row[col_ord_start], errors='coerce')
                d_e = pd.to_datetime(row[col_ord_end], errors='coerce')
                
                if pd.notnull(d_s) and start_range <= d_s <= end_range:
                    rev = clean_currency(row[col_ord_cost])
                    total_revenue += rev
                    
                    # Detailed Strings
                    start_str = f"{d_s.strftime('%Y-%m-%d %I:%M %p')} - {row[col_ord_loc_start]}"
                    end_str = f"{d_e.strftime('%Y-%m-%d %I:%M %p')} - {row[col_ord_loc_end]}" if pd.notnull(d_e) else "-"
                    
                    trips_data.append({
                        "السيارة": [k for k, v in car_options.items() if v == cid][0],
                        "رقم الطلب": row[col_ord_id],
                        "بداية الرحلة": start_str,
                        "نهاية الرحلة": end_str,
                        "الإيراد": format_egp(rev)
                    })

    # Expense Columns
    col_exp_car = get_col_by_letter(df_car_exp, 'S')
    col_exp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_exp_y = get_col_by_letter(df_car_exp, 'Y')
    col_exp_m = get_col_by_letter(df_car_exp, 'X')
    col_exp_d = get_col_by_letter(df_car_exp, 'W')
    
    # Detail logic based on prompt
    col_exp_type_ar = get_col_by_letter(df_car_exp, 'E') # Type (Arabic)
    col_exp_maint_ar = get_col_by_letter(df_car_exp, 'H') # Maint Statement (Arabic) -- FIXED to H
    col_exp_stmt_ar = get_col_by_letter(df_car_exp, 'K') # General Statement
    
    col_ref_q = get_col_by_letter(df_car_exp, 'Q')
    col_ref_r = get_col_by_letter(df_car_exp, 'R')
    col_ref_t = get_col_by_letter(df_car_exp, 'T') # Order Ref
    col_ref_s = get_col_by_letter(df_car_exp, 'S') # Car (Owner)

    if col_exp_car:
        for _, row in df_car_exp.iterrows():
            cid = clean_id_tag(row[col_exp_car])
            if cid in selected_ids:
                try:
                    y, m = int(clean_currency(row[col_exp_y])), int(clean_currency(row[col_exp_m]))
                    d_val = int(clean_currency(row[col_exp_d]))
                    
                    valid = False
                    if period_type=="سنة" and y==sel_year: valid=True
                    elif period_type=="شهر" and y==sel_year and m==sel_spec: valid=True
                    elif period_type=="ربع سنوي":
                         if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: valid=True
                    
                    if valid:
                        amt = clean_currency(row[col_exp_amt])
                        type_str = str(row[col_exp_type_ar]).strip() # e.g. تعاقد, صيانة
                        
                        # Logic for Display Name
                        display_name = type_str
                        is_maint = False
                        
                        if "صيانات" in type_str or "Maintenance" in type_str:
                            is_maint = True
                            display_name = str(row[col_exp_maint_ar]) # Col H
                        
                        elif "تعاقد" in type_str or "Contracting" in type_str:
                            display_name = f"{type_str} / {row[col_ref_q]} - {row[col_ref_r]}"
                            
                        elif "رد تامين" in type_str or "Deposit Refund" in type_str:
                            display_name = f"{type_str} / {row[col_ref_t]}"
                            
                        elif "عمولة وسيط" in type_str or "Brokerage" in type_str:
                            display_name = f"{type_str} / {row[col_ref_s]}"
                            
                        elif "تشغيل" in type_str or "Operating" in type_str or "مكتب" in type_str or "Office" in type_str:
                            display_name = f"{type_str} / {row[col_exp_stmt_ar]} - {row[col_ref_t]}"
                        
                        else:
                            # Standard Fallback
                            stmt = str(row[col_exp_stmt_ar]) if pd.notnull(row[col_exp_stmt_ar]) else ""
                            display_name = f"{type_str} - {stmt}"

                        entry = {
                            "السيارة": [k for k, v in car_options.items() if v == cid][0],
                            "التاريخ": f"{y}-{m:02d}-{d_val:02d}",
                            "البند": display_name,
                            "التكلفة": format_egp(amt)
                        }
                        
                        if is_maint: 
                            maint_list.append(entry)
                            total_maint += amt
                        else: 
                            exp_list.append(entry)
                            total_exp += amt
                except: continue

    if show_active and not trips_data: st.warning("لا يوجد بيانات."); return

    k1, k2 = st.columns(2)
    k1.metric("الإيراد الكلي", format_egp(total_revenue))
    k2.metric("الصيانة", format_egp(total_maint), delta_color="inverse")
    k3, k4 = st.columns(2)
    k3.metric("المصروفات", format_egp(total_exp), delta_color="inverse")
    k4.metric("الصافي", format_egp(total_revenue - total_maint - total_exp))
    
    t1, t2, t3 = st.tabs(["الرحلات", "الصيانة", "المصروفات"])
    with t1:
        if trips_data: st.dataframe(pd.DataFrame(trips_data), use_container_width=True)
        else: st.info("فارغ")
    with t2:
        if maint_list: st.dataframe(pd.DataFrame(maint_list), use_container_width=True)
        else: st.info("فارغ")
    with t3:
        if exp_list: st.dataframe(pd.DataFrame(exp_list), use_container_width=True)
        else: st.info("فارغ")

# --- 7. MODULE 3: CRM ---
def show_crm(dfs):
    st.title("👥 إدارة العملاء")
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
                
                status = "مكتمل"
                days = 0
                if pd.notnull(s) and pd.notnull(e):
                    days = (e - s).days
                    if s <= datetime.now() <= e: status = "نشط"
                    elif s > datetime.now(): status = "قادم"
                
                daily_rate = (amt / days) if days > 0 else 0
                rec['Spend'] += amt
                rec['Trips'] += 1
                rec['History'].append({
                    "رقم الطلب": row[col_ord_id],
                    "السيارة": car_display_map.get(cid, f"Unknown ({cid})"),
                    "البدء": s.strftime("%Y-%m-%d") if pd.notnull(s) else "-",
                    "الانتهاء": e.strftime("%Y-%m-%d") if pd.notnull(e) else "-",
                    "الأيام": days,
                    "التكلفة": format_egp(amt),
                    "اليومية": format_egp(daily_rate),
                    "الحالة": status
                })
            except: continue

    search = st.text_input("🔍 بحث عن عميل", "")
    data_list = []
    for k, v in client_db.items():
        data_list.append({'Display': v['Display'], 'Spend': v['Spend'], 'Trips': v['Trips'], 'Key': v['Name']})
    
    df_crm = pd.DataFrame(data_list)
    if not df_crm.empty:
        df_crm = df_crm.sort_values('Spend', ascending=False)
        if search: df_crm = df_crm[df_crm['Display'].str.contains(search, case=False, na=False)]

        c1, c2, c3 = st.columns(3)
        c1.metric("إجمالي العملاء", len(client_db))
        c2.metric("الأكثر إنفاقاً", df_crm.iloc[0]['Display'])
        c3.metric("متوسط العميل", format_egp(df_crm['Spend'].mean()))

        st.divider()
        col_list, col_detail = st.columns([1, 2])
        
        with col_list:
            df_display = df_crm.copy()
            df_display['Spend'] = df_display['Spend'].apply(format_egp)
            df_display.columns = ['العميل', 'الإنفاق', 'رحلات', 'Key']
            selection = st.dataframe(df_display[['العميل', 'الإنفاق', 'رحلات']], use_container_width=True, height=500, on_select="rerun", selection_mode="single-row", hide_index=True)
        
        with col_detail:
            sel_idx = selection.selection.rows
            if sel_idx:
                client_key = df_display.iloc[sel_idx[0]]['Key']
                client_data = client_db[client_key]
                st.info(f"**{client_data['Display']}**")
                m1, m2 = st.columns(2)
                m1.metric("الإجمالي", format_egp(client_data['Spend']))
                m2.metric("عدد الرحلات", client_data['Trips'])
                
                if client_data['History']:
                    hist_df = pd.DataFrame(client_data['History'])
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)
                else: st.warning("لا يوجد سجل.")
            else: st.info("👈 اختر عميلاً.")

# --- 8. MODULE 4: FINANCIAL HQ (ADVANCED LEDGER) ---
def show_financial_hq(dfs):
    st.title("💰 الإدارة المالية")
    if not dfs: return

    df_coll = dfs['collections']
    df_exp = dfs['expenses']
    df_car_exp = dfs['car_expenses']
    df_cars = dfs['cars']
    df_orders = dfs['orders']

    with st.expander("🗓️ إعدادات الفترة", expanded=True):
        f1, f2 = st.columns(2)
        period_type = f1.selectbox("نوع العرض", ["شهر", "ربع سنوي", "سنة"], key='fin_p')
        sel_year = f2.selectbox("السنة المالية", [2024, 2025, 2026], index=2, key='fin_y')
        f3, f4 = st.columns(2)
        if period_type == "شهر": sel_spec = f3.selectbox("الشهر", range(1, 13), index=0, key='fin_m')
        elif period_type == "ربع سنوي": sel_spec = f3.selectbox("الربع", [1, 2, 3, 4], index=0, key='fin_q')
        else: sel_spec = 0

    start_date, end_date = get_date_filter_range(period_type, sel_year, sel_spec)
    
    # 1. Cash Flow
    inflow, cash_in = [], 0.0
    col_coll_amt = get_col_by_letter(df_coll, 'R')
    col_coll_y = get_col_by_letter(df_coll, 'Q')
    col_coll_m = get_col_by_letter(df_coll, 'P')
    
    if col_coll_amt:
        for _, row in df_coll.iterrows():
            try:
                y, m = int(clean_currency(row[col_coll_y])), int(clean_currency(row[col_coll_m]))
                valid = False
                if period_type=="سنة" and y==sel_year: valid=True
                elif period_type=="شهر" and y==sel_year and m==sel_spec: valid=True
                elif period_type=="ربع سنوي":
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
                if period_type=="سنة" and y==sel_year: valid=True
                elif period_type=="شهر" and y==sel_year and m==sel_spec: valid=True
                elif period_type=="ربع سنوي":
                    if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: valid=True
                if valid: cash_out += clean_currency(row[col_exp_amt])
            except: continue

    # 2. Car Expenses Logic
    col_cexp_amt = get_col_by_letter(df_car_exp, 'Z')
    col_cexp_y = get_col_by_letter(df_car_exp, 'Y')
    col_cexp_m = get_col_by_letter(df_car_exp, 'X')
    col_cexp_car = get_col_by_letter(df_car_exp, 'S')
    col_cexp_id_g = get_col_by_letter(df_car_exp, 'G') # Type ID 1..8
    col_cexp_due_q = get_col_by_letter(df_car_exp, 'Q') # Due Month
    col_cexp_due_r = get_col_by_letter(df_car_exp, 'R') # Due Year

    deductions_in_period = {} 
    payments_to_owner_period = {} 
    payments_to_owner_lifetime = {} 

    if col_cexp_amt:
        for _, row in df_car_exp.iterrows():
            try:
                amt = clean_currency(row[col_cexp_amt])
                cid = clean_id_tag(row[col_cexp_car])
                type_id = str(row[col_cexp_id_g]).strip()
                
                # Global Period Check
                y, m = int(clean_currency(row[col_cexp_y])), int(clean_currency(row[col_cexp_m]))
                is_in_period = False
                if period_type=="سنة" and y==sel_year: is_in_period=True
                elif period_type=="شهر" and y==sel_year and m==sel_spec: is_in_period=True
                elif period_type=="ربع سنوي":
                    if y==sel_year and m in {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[sel_spec]: is_in_period=True
                
                if is_in_period: cash_out += amt

                # --- LEDGER LOGIC ---
                # 1. Contracting Payment (ID 1)
                # Check Due Month/Year in Q/R
                if type_id == '1': 
                    # If this specific contracting payment is meant for THIS period
                    pay_q = int(clean_currency(row[col_cexp_due_q]))
                    pay_r = int(clean_currency(row[col_cexp_due_r]))
                    
                    is_payment_for_this_period = False
                    if period_type == "شهر" and pay_r == sel_year and pay_q == sel_spec: is_payment_for_this_period = True
                    elif period_type == "سنة" and pay_r == sel_year: is_payment_for_this_period = True
                    
                    payments_to_owner_lifetime[cid] = payments_to_owner_lifetime.get(cid, 0) + amt
                    if is_payment_for_this_period: 
                        payments_to_owner_period[cid] = payments_to_owner_period.get(cid, 0) + amt
                
                elif type_id == '8': # Brokerage
                    payments_to_owner_lifetime[cid] = payments_to_owner_lifetime.get(cid, 0) + amt
                    if is_in_period: payments_to_owner_period[cid] = payments_to_owner_period.get(cid, 0) + amt

                elif type_id in ['3', '4']: # Fines(3), Maint(4) -> Deductions
                    if is_in_period:
                        deductions_in_period[cid] = deductions_in_period.get(cid, 0) + amt

            except: continue

    # 3. Ledger Building
    owner_ledger = []
    total_owner_payouts_due = 0.0 

    col_code = get_col_by_letter(df_cars, 'A')
    col_status = get_col_by_letter(df_cars, 'AZ')
    col_contract_start = get_col_by_letter(df_cars, 'AW')
    col_monthly_fee = get_col_by_letter(df_cars, 'CJ')
    col_pay_freq = get_col_by_letter(df_cars, 'CK') 
    col_deduct_pct = get_col_by_letter(df_cars, 'CL') 
    col_brokerage = get_col_by_letter(df_cars, 'CM') 

    for _, car in df_cars.iterrows():
        try:
            if col_status and not any(x in str(car[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
            cid = clean_id_tag(car[col_code])
            
            base_fee = clean_currency(car[col_monthly_fee])
            freq_days = clean_currency(car[col_pay_freq])
            if freq_days == 0: freq_days = 30 
            
            deduct_pct = clean_currency(car[col_deduct_pct])
            brokerage = clean_currency(car[col_brokerage])
            
            s_date = pd.to_datetime(car[col_contract_start], errors='coerce')
            if pd.isna(s_date): s_date = datetime(2023, 1, 1)
            
            try: due_day = datetime(sel_year, sel_spec if period_type=="شهر" else 1, s_date.day)
            except: due_day = datetime(sel_year, sel_spec, 28)
            due_date_display = due_day.strftime("%Y-%m-%d") if period_type=="شهر" else "متعدد"

            days_in_view = 30
            if period_type == "ربع سنوي": days_in_view = 90
            elif period_type == "سنة": days_in_view = 365
            
            cycles = days_in_view / freq_days
            gross_due = base_fee * cycles
            ops_fee_deduction = gross_due * (deduct_pct / 100)
            maint_deduction = deductions_in_period.get(cid, 0)
            
            net_due_period = gross_due - ops_fee_deduction + brokerage - maint_deduction
            total_owner_payouts_due += net_due_period

            days_active = (datetime.now() - s_date).days
            if days_active < 0: days_active = 0
            
            total_cycles_lifetime = days_active / freq_days
            lifetime_gross = base_fee * total_cycles_lifetime
            lifetime_ops_fee = lifetime_gross * (deduct_pct / 100)
            lifetime_brokerage = brokerage * (days_active / 30)
            
            lifetime_accrued = lifetime_gross - lifetime_ops_fee + lifetime_brokerage
            lifetime_paid = payments_to_owner_lifetime.get(cid, 0)
            balance = lifetime_accrued - lifetime_paid

            owner_ledger.append({
                "رقم السجل": cid, 
                "السيارة": f"{car[get_col_by_letter(df_cars, 'B')]} {car[get_col_by_letter(df_cars, 'E')]}",
                "تاريخ الاستحقاق": due_date_display,
                "إجمالي المستحق": format_egp(gross_due),
                "خصومات": format_egp(maint_deduction + ops_fee_deduction),
                "الصافي المستحق": net_due_period, 
                "تم دفعه": payments_to_owner_period.get(cid, 0),
                "الرصيد التراكمي": balance
            })
        except: continue

    tab1, tab2, tab3 = st.tabs(["التدفق النقدي", "الأرباح والخسائر", "كشف حساب الملاك"])
    
    with tab1:
        net = cash_in - cash_out
        c1, c2 = st.columns(2)
        c1.metric("وارد", format_egp(cash_in))
        c2.metric("صادر", format_egp(cash_out), delta_color="inverse")
        st.metric("السيولة الصافية", format_egp(net))
        fig = go.Figure(go.Waterfall(measure=["relative", "relative", "total"], x=["وارد", "صادر", "الصافي"], y=[cash_in, -cash_out, 0]))
        fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rev = sum(x['Amount'] for x in inflow if x['Category'] == "Revenue")
        profit = rev - cash_out - total_owner_payouts_due
        mrg = (profit/rev*100) if rev>0 else 0
        c1, c2 = st.columns(2)
        c1.metric("إيرادات التشغيل", format_egp(rev)); c2.metric("صافي الربح", format_egp(profit), f"{mrg:.1f}%")
        
        b1, b2 = st.columns(2)
        with b1: 
            fig = px.pie(names=["مصروفات", "مستحقات ملاك", "ربح"], values=[cash_out, total_owner_payouts_due, max(0, profit)], hole=0.5)
            fig.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with b2:
            fig = go.Figure(data=[go.Bar(name='إيراد', x=['P&L'], y=[rev]), go.Bar(name='تكلفة', x=['P&L'], y=[cash_out+total_owner_payouts_due])])
            fig.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if owner_ledger:
            df_l = pd.DataFrame(owner_ledger)
            df_l = df_l.sort_values('تاريخ الاستحقاق')
            for c in ["الصافي المستحق", "تم دفعه", "الرصيد التراكمي"]: df_l[c] = df_l[c].apply(format_egp)
            def highlight_balance(val):
                color = 'white'
                try:
                    num = float(str(val).replace(' ج.م','').replace(',',''))
                    if num > 100: color = '#ff4b4b' 
                    elif num < -100: color = '#00c853'
                except: pass
                return f'color: {color}'
            st.dataframe(df_l.style.map(highlight_balance, subset=['الرصيد التراكمي']), use_container_width=True, height=500)
        else: st.info("لا توجد عقود نشطة.")

# --- 9. MODULE 5: RISK RADAR ---
def show_risk_radar(dfs):
    st.title("⚠️ رادار المخاطر")
    if not dfs: return
    
    df_cars = dfs['cars']
    today = datetime.now()
    
    risks = {'License': [], 'Insurance': [], 'Contract': []}
    
    col_lic_end = get_col_by_letter(df_cars, 'AQ') 
    col_exam_end = get_col_by_letter(df_cars, 'BD')
    col_lic_status = get_col_by_letter(df_cars, 'AT')
    col_ins_end = get_col_by_letter(df_cars, 'BJ')
    col_ins_status = get_col_by_letter(df_cars, 'BN')
    col_con_end = get_col_by_letter(df_cars, 'BC')
    col_name = get_col_by_letter(df_cars, 'B')
    col_model = get_col_by_letter(df_cars, 'E')
    col_code = get_col_by_letter(df_cars, 'A') 
    col_plate = get_col_by_letter(df_cars, 'AC') 
    col_status = get_col_by_letter(df_cars, 'AZ')
    plate_cols = ['AC','AB','AA','Z','Y','X','W']

    for _, row in df_cars.iterrows():
        try:
            if col_status and not any(x in str(row[col_status]) for x in ['Valid', 'Active', 'ساري']): continue
            cid = clean_id_tag(row[col_code])
            cname = f"[{cid}] {row[col_name]} {row[col_model]}"
            plate = "".join([str(row[get_col_by_letter(df_cars, p)]) + " " for p in plate_cols if pd.notnull(row[get_col_by_letter(df_cars, p)])]).strip()
            
            # LICENSE
            lic_valid = True
            if col_lic_status: lic_valid = any(x in str(row[col_lic_status]) for x in ['Valid', 'Active', 'ساري'])
            if lic_valid:
                d_lic = pd.to_datetime(row[col_lic_end], errors='coerce') if col_lic_end else None
                d_exam = pd.to_datetime(row[col_exam_end], errors='coerce') if col_exam_end else None
                target, reason = None, "ترخيص"
                if d_lic and d_exam:
                    if d_lic == d_exam: target, reason = d_lic, "ترخيص + فحص"
                    elif d_lic < d_exam: target, reason = d_lic, "ترخيص"
                    else: target, reason = d_exam, "فحص"
                elif d_lic: target, reason = d_lic, "ترخيص"
                elif d_exam: target, reason = d_exam, "فحص"
                if target:
                    days = (target - today).days
                    bucket = None
                    if days <= 90: bucket = "خطر مرتفع (0-3 أشهر)"
                    elif days <= 180: bucket = "خطر متوسط (3-6 أشهر)"
                    elif days > 180: bucket = "خطر منخفض (> 6 أشهر)"
                    if bucket: risks['License'].append({'السيارة': cname, 'اللوحة': plate, 'السبب': reason, 'الاستحقاق': target.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

            # INSURANCE
            has_ins = False
            if col_ins_status:
                s_val = str(row[col_ins_status]).lower()
                if "yes" in s_val or "يوجد" in s_val: has_ins = True
            if has_ins and col_ins_end:
                d = pd.to_datetime(row[col_ins_end], errors='coerce')
                if pd.notnull(d):
                    days = (d - today).days
                    bucket = None
                    if days <= 90: bucket = "خطر مرتفع (0-3 أشهر)"
                    elif days <= 180: bucket = "خطر متوسط (3-6 أشهر)"
                    elif days > 180: bucket = "خطر منخفض (> 6 أشهر)"
                    if bucket: risks['Insurance'].append({'السيارة': cname, 'اللوحة': plate, 'الاستحقاق': d.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

            # CONTRACT
            if col_con_end:
                d = pd.to_datetime(row[col_con_end], errors='coerce')
                if pd.notnull(d):
                    days = (d - today).days
                    bucket = None
                    if days <= 90: bucket = "خطر مرتفع (0-3 أشهر)"
                    elif days <= 180: bucket = "خطر متوسط (3-6 أشهر)"
                    elif days > 180: bucket = "خطر منخفض (> 6 أشهر)"
                    if bucket: risks['Contract'].append({'السيارة': cname, 'اللوحة': plate, 'الاستحقاق': d.strftime("%Y-%m-%d"), 'التصنيف': bucket, 'Days': days})

        except: continue

    t1, t2, t3 = st.tabs(["📄 الترخيص", "🛡️ التأمين", "📝 العقود"])
    def render_tab(category):
        items = risks[category]
        if not items:
            st.success("✅ الكل سليم.")
            return
        df = pd.DataFrame(items).sort_values('Days')
        b1 = df[df['التصنيف'] == "خطر مرتفع (0-3 أشهر)"]
        b2 = df[df['التصنيف'] == "خطر متوسط (3-6 أشهر)"]
        b3 = df[df['التصنيف'] == "خطر منخفض (> 6 أشهر)"]
        with st.expander(f"🔴 خطر مرتفع (0-3 أشهر) [{len(b1)}]", expanded=True):
            if not b1.empty: st.dataframe(b1.drop(columns=['التصنيف', 'Days']), use_container_width=True)
            else: st.info("لا يوجد")
        with st.expander(f"🟡 خطر متوسط (3-6 أشهر) [{len(b2)}]", expanded=False):
            if not b2.empty: st.dataframe(b2.drop(columns=['التصنيف', 'Days']), use_container_width=True)
            else: st.info("لا يوجد")
        with st.expander(f"🟢 خطر منخفض (> 6 أشهر) [{len(b3)}]", expanded=False):
            if not b3.empty: st.dataframe(b3.drop(columns=['التصنيف', 'Days']), use_container_width=True)
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
