import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
from googleapiclient.discovery import build
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re
import string  # <--- FIXED: Added missing import
from datetime import timedelta, datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental Command Center", layout="wide", page_icon="🚘")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    
    /* Metrics Styling */
    .stMetric { 
        background-color: #ffffff !important; 
        border-radius: 12px; 
        padding: 15px; 
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #6c757d !important; font-size: 0.9rem; }
    [data-testid="stMetricValue"] { color: #212529 !important; font-weight: 700; font-size: 1.6rem; }
    
    /* Table Styling */
    .stDataFrame { direction: ltr; }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPERS ---
def excel_col_to_index(col_str):
    """ Converts Excel Column Letter (e.g., 'BA', 'CJ') to zero-based index """
    num = 0
    for c in col_str:
        if c.upper() in string.ascii_uppercase: # <--- FIXED: Using string module
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1

def clean_money(x):
    if pd.isna(x) or str(x).strip() == '': return 0.0
    s = str(x).replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

# --- 3. DATA LOADER ---
@st.cache_data(ttl=600)
def load_data():
    if "gcp_service_account" not in st.secrets:
        st.error("Missing Secrets.")
        return None

    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)

    def get_sheet(sheet_id, range_name, header_row=0):
        try:
            # We fetch A:ZZ to ensure we get all far columns like CJ
            res = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = res.get('values', [])
            if not vals: return pd.DataFrame()
            
            # Use specific header row
            if len(vals) > header_row:
                headers = vals[header_row]
                data = vals[header_row+1:]
                
                # Normalize
                max_len = len(headers)
                clean_data = [row[:max_len] + [None]*(max_len-len(row)) for row in data]
                return pd.DataFrame(clean_data, columns=headers)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading sheet: {e}")
            return pd.DataFrame()

    # IDS
    ids = {
        'coll': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg",
        'gen': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_exp': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'orders': "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A",
        'cars': "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg",
        'clients': "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg"
    }

    dfs = {}
    dfs['coll'] = get_sheet(ids['coll'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
    dfs['gen'] = get_sheet(ids['gen'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
    dfs['car_exp'] = get_sheet(ids['car_exp'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
    dfs['orders'] = get_sheet(ids['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 0)
    dfs['cars'] = get_sheet(ids['cars'], "'قاعدة البيانات'!A:ZZ", 0)
    dfs['clients'] = get_sheet(ids['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)

    return dfs

# --- 4. PROCESSING LOGIC ---
dfs = load_data()

if dfs:
    # --- TIME MACHINE (SIDEBAR) ---
    st.sidebar.title("🔍 Time Filter")
    sel_year = st.sidebar.selectbox("Year", [2024, 2025, 2026, 2027], index=2) # Default 2026
    sel_month = st.sidebar.selectbox("Month", range(1, 13), index=0) # Default Jan
    
    # --- A. PROCESS CARS (THE ENGINE) ---
    df_cars_raw = dfs['cars']
    cars_clean = []
    
    if not df_cars_raw.empty:
        raw_values = df_cars_raw.values.tolist()
        
        for row in raw_values:
            # Safe getter helper
            def get(col_letter):
                idx = excel_col_to_index(col_letter)
                if idx < len(row): return row[idx]
                return None

            code = str(get('A') or '').strip()
            if not code or code == 'None': continue

            # 1. Construct Name (Type B + Model E + Year H + Color J + Seats O)
            c_name = f"{get('B') or ''} {get('E') or ''} ({get('H') or ''}) - {get('J') or ''} - {get('O') or ''} Seats"
            
            # 2. Construct Plate (AC, AB, AA, Z, Y, X, W)
            plate_parts = [get('AC'), get('AB'), get('AA'), get('Z'), get('Y'), get('X'), get('W')]
            plate = " ".join([str(p) for p in plate_parts if p])
            
            # 3. Status (BA)
            status_raw = str(get('BA') or '')
            is_active = 'ساري' in status_raw or 'Valid' in status_raw or 'valid' in status_raw
            
            # 4. Dates & KM
            km_start = clean_money(get('AV'))
            lic_end = pd.to_datetime(get('AQ'), errors='coerce')
            ins_end = pd.to_datetime(get('BK'), errors='coerce')
            contract_end = pd.to_datetime(get('AX'), errors='coerce')
            
            # 5. Owner Payment Logic
            pay_amt = clean_money(get('CJ'))
            pay_freq = clean_money(get('CK'))
            pay_start = pd.to_datetime(get('CL'), errors='coerce')
            
            cars_clean.append({
                'Code': code,
                'Name': c_name,
                'Plate': plate,
                'Active': is_active,
                'KM_Start': km_start,
                'License_End': lic_end,
                'Insurance_End': ins_end,
                'Contract_End': contract_end,
                'Pay_Amount': pay_amt,
                'Pay_Freq': int(pay_freq) if pay_freq > 0 else 45,
                'Pay_Start': pay_start
            })
    
    df_cars = pd.DataFrame(cars_clean)
    
    # --- B. PROCESS FINANCIALS & FILTERS ---
    
    # 1. Filter Financials by Month
    def filter_month(df):
        if df.empty: return df
        y_col = next((c for c in df.columns if 'سنة' in c or 'Year' in c), None)
        m_col = next((c for c in df.columns if 'شهر' in c or 'Month' in c), None)
        if y_col and m_col:
            # Filter
            return df[
                (df[y_col].astype(str).str.contains(str(sel_year))) & 
                (df[m_col].astype(str).str.contains(str(sel_month)))
            ]
        return df

    df_coll_m = filter_month(dfs['coll'])
    df_gen_m = filter_month(dfs['gen'])
    df_car_exp_m = filter_month(dfs['car_exp'])
    
    # Clean Money
    for d in [df_coll_m, df_gen_m, df_car_exp_m]:
        for c in d.columns:
            if 'قيمة' in c: d[c] = d[c].apply(clean_money)

    # 2. Calculate Owner Liabilities for Selected Month
    owner_liabilities = []
    total_owner_fees = 0
    
    for _, car in df_cars.iterrows():
        # Only calculate if Active and has Payment info
        if not car['Active'] or pd.isna(car['Pay_Start']) or car['Pay_Amount'] == 0:
            continue
            
        # Generate Schedule
        curr = car['Pay_Start']
        # End date fallback
        end = car['Contract_End'] if pd.notnull(car['Contract_End']) else datetime(2030, 1, 1)
        
        while curr <= end:
            # Check if this occurrence falls in selected month
            if curr.year == sel_year and curr.month == sel_month:
                # AUDIT: Did we pay this?
                status = "PENDING"
                match_row = ""
                
                if not df_car_exp_m.empty:
                    exp_code_col = next((c for c in df_car_exp_m.columns if 'كود السيارة' in c), None)
                    exp_val_col = next((c for c in df_car_exp_m.columns if 'قيمة' in c), None)
                    
                    if exp_code_col and exp_val_col:
                        # Filter for car code
                        matches = df_car_exp_m[df_car_exp_m[exp_code_col].astype(str).str.strip() == car['Code']]
                        for idx, m in matches.iterrows():
                            # Check amount (tolerance 10%)
                            if abs(clean_money(m[exp_val_col]) - car['Pay_Amount']) < (car['Pay_Amount'] * 0.1):
                                status = "PAID"
                                match_row = f"Row {idx+2}"
                                break
                
                owner_liabilities.append({
                    'Due Date': curr.strftime('%Y-%m-%d'),
                    'Car Name': car['Name'],
                    'Code': car['Code'],
                    'Amount': car['Pay_Amount'],
                    'Status': status,
                    'Note': match_row
                })
                total_owner_fees += car['Pay_Amount']
                
            curr += timedelta(days=car['Pay_Freq'])

    df_liab = pd.DataFrame(owner_liabilities)

    # 3. Profit Calculation
    rev = df_coll_m['قيمة التحصيل'].sum() if not df_coll_m.empty else 0
    exp_ops = df_gen_m['قيمة المصروف'].sum() if not df_gen_m.empty else 0
    exp_maint = df_car_exp_m['قيمة المصروف'].sum() if not df_car_exp_m.empty else 0
    
    # We show Cash Profit (Actual) but alert about Liabilities
    total_actual_exp = exp_ops + exp_maint
    cash_profit = rev - total_actual_exp
    
    # --- DASHBOARD UI ---
    st.title(f"📊 Dashboard: {sel_month} / {sel_year}")
    
    # KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cash Profit (Actual)", f"{cash_profit:,.0f} EGP", delta="Cash Flow")
    k2.metric("Revenue", f"{rev:,.0f} EGP")
    k3.metric("Expenses (Ops + Maint)", f"{total_actual_exp:,.0f} EGP", delta_color="inverse")
    k4.metric("Owner Fees Due", f"{total_owner_fees:,.0f} EGP", delta_color="off")

    st.markdown("---")

    # TABS
    tab_fleet, tab_owners, tab_fin, tab_alert, tab_ai = st.tabs([
        "🚗 Fleet (الأسطول)", "🤝 Owner Payments (مستحقات)", "📉 Financials (المالية)", "⚠️ Alerts (تنبيهات)", "🧠 AI"
    ])

    with tab_fleet:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Active Fleet Status")
            active = df_cars[df_cars['Active'] == True]
            inactive = df_cars[df_cars['Active'] == False]
            
            # Sort Inactive by Contract End (Latest to Oldest) as requested
            if not inactive.empty:
                inactive = inactive.sort_values(by='Contract_End', ascending=False)

            st.success(f"Active Cars: {len(active)}")
            st.dataframe(
                active[['Code', 'Name', 'Plate', 'KM_Start', 'Contract_End']],
                use_container_width=True
            )
            
            if not inactive.empty:
                st.markdown("### ❌ Inactive / Expired Contracts")
                st.dataframe(inactive[['Code', 'Name', 'Contract_End', 'Pay_Amount']], use_container_width=True)

        with c2:
            st.subheader("KM Tracker")
            st.dataframe(active[['Code', 'KM_Start']], use_container_width=True)

    with tab_owners:
        st.subheader("Owner Payment Schedule (This Month)")
        if not df_liab.empty:
            def color_status(val):
                color = '#d4edda' if val == 'PAID' else '#f8d7da'
                return f'background-color: {color}; color: black'
            
            st.dataframe(
                df_liab.style.applymap(color_status, subset=['Status']),
                use_container_width=True
            )
        else:
            st.success("No owner payments due this month.")

    with tab_fin:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Revenue vs Expenses")
            fig = px.bar(x=['Revenue', 'Expenses', 'Owner Liability'], 
                         y=[rev, total_actual_exp, total_owner_fees],
                         color=['Rev', 'Exp', 'Liab'],
                         color_discrete_sequence=['green', 'red', 'orange'])
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Expense Breakdown")
            items = []
            if not df_gen_m.empty:
                col = next((c for c in df_gen_m.columns if 'بيان' in c), 'Item')
                items.append(df_gen_m[[col, 'قيمة المصروف']].rename(columns={col:'Item', 'قيمة المصروف':'Value'}))
            
            if not df_car_exp_m.empty:
                col = next((c for c in df_car_exp_m.columns if 'نوع' in c), 'Item')
                items.append(df_car_exp_m[[col, 'قيمة المصروف']].rename(columns={col:'Item', 'قيمة المصروف':'Value'}))
            
            if items:
                full_exp = pd.concat(items)
                if not full_exp.empty and full_exp['Value'].sum() > 0:
                    fig2 = px.treemap(full_exp, path=['Item'], values='Value')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No expenses recorded.")

    with tab_alert:
        c1, c2 = st.columns(2)
        today = datetime.today()
        limit_3m = today + timedelta(days=90)
        limit_6m = today + timedelta(days=180)
        
        alerts_lic = []
        alerts_ins = []
        
        for _, car in df_cars.iterrows():
            if not car['Active']: continue 
            
            # License
            if pd.notnull(car['License_End']):
                if today < car['License_End'] <= limit_3m:
                    alerts_lic.append({'Car': car['Name'], 'Date': car['License_End'], 'Priority': 'HIGH (3 Mo)'})
                elif limit_3m < car['License_End'] <= limit_6m:
                    alerts_lic.append({'Car': car['Name'], 'Date': car['License_End'], 'Priority': 'Medium (6 Mo)'})
            
            # Insurance
            if pd.notnull(car['Insurance_End']):
                if today < car['Insurance_End'] <= limit_3m:
                    alerts_ins.append({'Car': car['Name'], 'Date': car['Insurance_End'], 'Priority': 'HIGH (3 Mo)'})
                elif limit_3m < car['Insurance_End'] <= limit_6m:
                    alerts_ins.append({'Car': car['Name'], 'Date': car['Insurance_End'], 'Priority': 'Medium (6 Mo)'})
        
        with c1:
            st.subheader("📄 License Expiry")
            if alerts_lic: st.table(pd.DataFrame(alerts_lic))
            else: st.success("No licenses expiring soon.")
                
        with c2:
            st.subheader("🛡️ Insurance Expiry")
            if alerts_ins: st.table(pd.DataFrame(alerts_ins))
            else: st.success("No insurance expiring soon.")

    with tab_ai:
        if st.button("Generate AI Briefing"):
            if 'GOOGLE_API_KEY' not in st.secrets:
                st.error("Missing Google API Key")
            else:
                with st.spinner("Analyzing..."):
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                    
                    prompt = f"""
                    Role: Business Consultant for Egypt Car Rental.
                    Data for {sel_month}/{sel_year}:
                    - Revenue: {rev}
                    - Cash Expenses: {total_actual_exp}
                    - Owner Liabilities: {total_owner_fees}
                    - Cash Profit: {cash_profit}
                    - Active Cars: {len(active)}
                    
                    Task: Write a professional executive summary in Arabic.
                    """
                    advisor = Agent(role='Advisor', goal='Analysis', backstory='Expert', llm=llm)
                    task = Task(description=prompt, agent=advisor, expected_output="Briefing")
                    crew = Crew(agents=[advisor], tasks=[task])
                    st.markdown(crew.kickoff())
