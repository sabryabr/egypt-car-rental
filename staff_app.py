import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import string

# --- 1. APP CONFIG (OPERATIONS VIEW) ---
st.set_page_config(page_title="Brothers Ops", layout="wide", page_icon="🛠️", initial_sidebar_state="expanded")

# --- 2. STYLING (RTL & CLEAN) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    .main { direction: rtl; font-family: 'Cairo', sans-serif; text-align: right; }
    h1, h2, h3, h4, h5, .stMarkdown, .stButton button, .stSelectbox, .stTextInput, .stDateInput { 
        font-family: 'Cairo', sans-serif; text-align: right; 
    }
    .stDataFrame { direction: ltr; }
    /* Hide default Streamlit menu for staff */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. DATA CONNECTION (READ ONLY) ---
@st.cache_resource
def get_service():
    if "gcp_service_account" not in st.secrets:
        st.error("⚠️ Secrets Missing!")
        return None
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    return build('sheets', 'v4', credentials=creds)

@st.cache_data(ttl=600) # Cache for 10 mins
def load_data():
    service = get_service()
    if not service: return None
    
    IDS = {
        'cars': "1tQVkPj7tCnrKsHEIs04a1WzzC04jpOWuLsXgXOkVMkk",
        'orders': "1T6j2xnRBTY31crQcJHioKurs4Rvaj-VlEQkm6joGxGM",
        'clients': "13YZOGdRCEy7IMZHiTmjLFyO417P8dD0m5Sh9xwKI8js"
    }

    def fetch(id, range_name):
        try:
            result = service.spreadsheets().values().get(spreadsheetId=id, range=range_name).execute()
            vals = result.get('values', [])
            if len(vals) > 1:
                return pd.DataFrame(vals[1:], columns=vals[0])
            return pd.DataFrame()
        except: return pd.DataFrame()

    with st.spinner("جاري تحديث بيانات الأسطول..."):
        return {
            'cars': fetch(IDS['cars'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ"),
            'orders': fetch(IDS['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ"),
            'clients': fetch(IDS['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ")
        }

# --- 4. HELPERS ---
def get_col(df, letter):
    idx = 0
    for char in letter: idx = idx * 26 + (ord(char.upper()) - ord('A')) + 1
    idx -= 1
    return df.columns[idx] if idx < len(df.columns) else None

def clean_str(x): return str(x).strip() if pd.notnull(x) else ""

# --- 5. MODULE: CONTRACT GENERATOR ---
def show_contract_generator(dfs):
    st.title("📝 إصدار العقود (Contract System)")
    st.caption("يتم استخدام البيانات الحالية لملء العقد تلقائياً")
    
    df_cars = dfs['cars']
    
    # A. PREPARE CAR LIST
    car_options = {}
    col_code = get_col(df_cars, 'A')
    col_make = get_col(df_cars, 'B')
    col_model = get_col(df_cars, 'E')
    col_plate = get_col(df_cars, 'AC')
    col_km = get_col(df_cars, 'I') # Assuming I is KM

    if col_code:
        for _, row in df_cars.iterrows():
            try:
                # Filter Active Only
                status = str(row[get_col(df_cars, 'AZ')]).lower()
                if 'valid' in status or 'active' in status or 'ساري' in status:
                    lbl = f"{row[col_make]} {row[col_model]} - {row[col_plate]}"
                    car_options[lbl] = {
                        'Make': row[col_make], 'Model': row[col_model], 
                        'Plate': row[col_plate], 'KM': row[col_km]
                    }
            except: continue

    # B. INPUT FORM
    with st.form("gen_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 1. بيانات العميل")
            name = st.text_input("الاسم بالكامل (Full Name)")
            nat = st.text_input("الجنسية (Nationality)")
            phone = st.text_input("الهاتف (Phone)")
            pass_id = st.text_input("رقم الهوية/الباسبور")
            lic = st.text_input("رقم الرخصة")
            addr = st.text_input("العنوان")

        with c2:
            st.markdown("##### 2. بيانات السيارة")
            sel_car = st.selectbox("اختر السيارة", [""] + list(car_options.keys()))
            
            # Auto-fill KM and Plate if car selected
            def_plate, def_km = "", ""
            if sel_car and sel_car in car_options:
                def_plate = car_options[sel_car]['Plate']
                def_km = car_options[sel_car]['KM']

            plate = st.text_input("رقم اللوحة", value=def_plate)
            km = st.text_input("العداد الحالي (KM)", value=def_km)
            fuel = st.selectbox("الوقود", ["Full (8/8)", "3/4", "1/2", "1/4", "Empty"])
        
        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("##### 3. الفترة")
            p_date = st.date_input("تاريخ الاستلام", datetime.now())
            p_time = st.time_input("وقت الاستلام", datetime.now().time())
            d_date = st.date_input("تاريخ التسليم", datetime.now() + timedelta(days=1))
        
        with c4:
            st.markdown("##### 4. المالية")
            rent = st.number_input("إجمالي الإيجار", 0)
            dep = st.number_input("التأمين (Deposit)", 0)
            tier = st.selectbox("نوع التأمين", ["Basic (Client 100% Liable)", "Intermediate (70%)", "Full (0% Liable)"])
            method = st.selectbox("طريقة الدفع", ["Cash (EGP)", "Cash (USD)", "Credit Card"])

        submit = st.form_submit_button("🖨️ إنشاء العقد للطباعة")

    # C. GENERATE HTML
    if submit:
        # Format Data
        pickup_str = f"{p_date} {p_time.strftime('%H:%M')}"
        drop_str = f"{d_date}"
        
        # Prepare HTML (Using the Pro V6 Layout)
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Contract</title>
            <style>
                body {{ font-family: sans-serif; -webkit-print-color-adjust: exact; }}
                @media print {{ .no-print {{ display: none; }} }}
                .page {{ width: 210mm; min-height: 290mm; padding: 10mm; margin: auto; border: 1px solid #ccc; background: white; }}
                .header {{ display: flex; justify-content: space-between; border-bottom: 2px solid #04509D; padding-bottom: 10px; }}
                .box {{ border: 1px solid #000; padding: 5px; font-size: 11px; margin-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-bottom: 10px; }}
                th, td {{ border: 1px solid #000; padding: 4px; text-align: left; }}
                th {{ background: #eee; }}
                .btn {{ background: #04509D; color: white; padding: 15px; width: 100%; display: block; text-align: center; font-size: 18px; cursor: pointer; text-decoration: none; }}
            </style>
        </head>
        <body>
            <div class="no-print" style="margin-bottom: 20px;">
                <button onclick="window.print()" class="btn">🖨️ CLICK TO PRINT NOW</button>
            </div>

            <div class="page">
                <div class="header">
                    <img src="https://brothersegy.com/wp-content/uploads/2025/07/22wdd.jpg" height="60">
                    <div style="text-align:right; font-size:10px;">
                        <strong>EL ALAKHUA EL MUTAHIDUN</strong><br>
                        Tax: 589-464-418 | CR: 10770<br>
                        Hurghada, Red Sea | +20 10 92725181
                    </div>
                </div>
                
                <h2 style="text-align:center; margin:10px 0; font-size:16px;">RENTAL AGREEMENT</h2>

                <div style="display:flex; gap:10px;">
                    <div class="box" style="flex:1">
                        <strong>FIRST PARTY (LESSOR):</strong><br>
                        Name: El Alakhua El Mutahidun<br>
                        Tax Card: 589-464-418
                    </div>
                    <div class="box" style="flex:1">
                        <strong>SECOND PARTY (LESSEE):</strong><br>
                        Name: {name}<br>
                        ID/Passport: {pass_id}<br>
                        Nationality: {nat}<br>
                        Phone: {phone}<br>
                        Address: {addr}
                    </div>
                </div>

                <table>
                    <tr><th>Car Model</th><th>Plate</th><th>KM</th><th>Fuel</th></tr>
                    <tr><td>{sel_car.split('-')[0] if sel_car else ''}</td><td>{plate}</td><td>{km}</td><td>{fuel}</td></tr>
                </table>

                <table>
                    <tr><th>Pickup</th><th>Dropoff</th></tr>
                    <tr><td>{pickup_str} (Airport)</td><td>{drop_str} (Airport)</td></tr>
                </table>

                <table>
                    <tr><th>Rental Fee</th><th>Deposit</th><th>Insurance Tier</th></tr>
                    <tr><td><b>{rent}</b></td><td><b>{dep}</b></td><td><b>{tier}</b></td></tr>
                </table>

                <div style="border: 2px solid #c0392b; background: #fff5f5; color: #c0392b; padding: 10px; font-size: 10px; text-align: center; font-weight: bold;">
                    ⚠️ IMPORTANT: In case of ANY accident, a POLICE REPORT is MANDATORY. 
                    Without it, insurance is VOID and you are 100% liable.
                </div>

                <div style="font-size: 8px; column-count: 2; margin-top: 10px;">
                    1. <b>USAGE:</b> Personal use only. No Safari/Off-road. No Smoking.<br>
                    2. <b>FUEL:</b> Return same level. 120KM/day limit.<br>
                    3. <b>FINES:</b> Client pays all traffic fines.<br>
                    4. <b>DISPUTES:</b> Damage cost disputes settled by independent Service Center report (Binding).
                </div>

                <div style="display:flex; justify-content:space-between; margin-top:30px; border-top:1px solid #000; padding-top:10px;">
                    <div>Lessor Signature:<br><br>__________________</div>
                    <div>Lessee Signature:<br><br>__________________</div>
                </div>
            </div>
            
            <div class="page" style="page-break-before: always;">
                <h3 style="text-align:center; border-bottom:1px solid #000;">VEHICLE CONDITION</h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div style="border:1px solid #ccc; height:100px; position:relative;">
                        <span style="position:absolute; top:2px; left:2px; font-size:8px;">FRONT</span>
                        <svg width="100%" height="100%" viewBox="0 0 200 60"><path d="M40,50 L160,50 L180,30 L160,10 L40,10 L20,30 Z" fill="none" stroke="#999" stroke-width="2"/></svg>
                    </div>
                    <div style="border:1px solid #ccc; height:100px; position:relative;">
                        <span style="position:absolute; top:2px; left:2px; font-size:8px;">SIDE</span>
                        <svg width="100%" height="100%" viewBox="0 0 200 60"><rect x="10" y="20" width="180" height="30" fill="none" stroke="#999"/></svg>
                    </div>
                </div>
                <div style="font-size:10px; margin-top:5px;">Marking: X=Scratch, O=Dent. <br>Client Sign: _________________</div>

                <div style="border-top: 2px dashed #000; margin: 30px 0; text-align: center;">✂️ CUT HERE</div>

                <div style="border: 2px dashed #000; padding: 15px;">
                    <div style="display:flex; justify-content:space-between;">
                        <strong>OFFICIAL RECEIPT</strong>
                        <span>Date: {datetime.now().strftime('%Y-%m-%d')}</span>
                    </div>
                    <br>
                    Received From: <b>{name}</b><br>
                    Total Amount: <b>{rent} (Rent) + {dep} (Dep)</b><br>
                    Method: {method}<br>
                    <br>
                    <div style="text-align:right;">Received By: __________________</div>
                </div>
            </div>
        </body>
        </html>
        """
        components.html(html_code, height=800, scrolling=True)

# --- 6. MODULE: OPERATIONS (STAFF VIEW) ---
def show_ops(dfs):
    st.title("🏠 العمليات (Operations)")
    df_ord = dfs['orders']
    
    # Simple Metrics
    col_start = get_col(df_ord, 'L')
    col_end = get_col(df_ord, 'V')
    
    active = 0
    today = datetime.now()
    
    if col_start and col_end:
        for _, row in df_ord.iterrows():
            try:
                s = pd.to_datetime(row[col_start])
                e = pd.to_datetime(row[col_end])
                if s <= today <= e: active += 1
            except: continue
            
    st.metric("السيارات المؤجرة حالياً", active)
    
    st.subheader("جدول الحجوزات")
    # Show simplified table (No prices)
    if not df_ord.empty:
        # Create a clean view for staff
        cols_to_show = [get_col(df_ord, 'A'), get_col(df_ord, 'B'), get_col(df_ord, 'C'), col_start, col_end]
        valid_cols = [c for c in cols_to_show if c]
        st.dataframe(df_ord[valid_cols], use_container_width=True)

# --- 7. MAIN NAVIGATION ---
st.sidebar.title("Staff System")
page = st.sidebar.radio("القائمة", ["إصدار عقد", "العمليات", "ملف السيارات"])

dfs = load_data()

if page == "إصدار عقد": show_contract_generator(dfs)
elif page == "العمليات": show_ops(dfs)
elif page == "ملف السيارات": 
    st.title("🚗 أسطول السيارات")
    st.dataframe(dfs['cars'], use_container_width=True)
