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
from datetime import timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental CEO Dashboard", layout="wide", page_icon="🚘")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    .stMetric { 
        background-color: #ffffff !important; 
        border-radius: 12px; 
        padding: 15px; 
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #6c757d !important; font-size: 0.9rem; }
    [data-testid="stMetricValue"] { color: #212529 !important; font-weight: 700; font-size: 1.8rem; }
    .stDataFrame { direction: ltr; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e7f1ff;
        color: #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPERS ---
def clean_money(x):
    if pd.isna(x) or x == '': return 0.0
    s = str(x).replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

def parse_date(df, col_name):
    """ Tries to parse dates flexibly """
    if col_name in df.columns:
        return pd.to_datetime(df[col_name], errors='coerce')
    return pd.Series([pd.NaT]*len(df))

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

    def get_sheet(sheet_id, range_name):
        try:
            res = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = res.get('values', [])
            if not vals: return pd.DataFrame()
            
            # Smart Header Search
            header_idx = 0
            keywords = ['No.', 'كود', 'Name', 'Code', 'Date', 'Type']
            for i, row in enumerate(vals[:10]):
                if any(k in str(row) for k in keywords):
                    header_idx = i; break
            
            headers = [str(h).strip() for h in vals[header_idx]]
            # Uniquify headers
            seen = {}
            unique_headers = []
            for h in headers:
                if h in seen:
                    seen[h] += 1
                    unique_headers.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    unique_headers.append(h)
            
            data = vals[header_idx+1:]
            max_len = len(unique_headers)
            clean_data = [row[:max_len] + [None]*(max_len-len(row)) for row in data]
            
            return pd.DataFrame(clean_data, columns=unique_headers)
        except Exception: return pd.DataFrame()

    # IDs
    ids = {
        'coll': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg",
        'gen': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_exp': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'orders': "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A",
        'cars': "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg",
        'clients': "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg"
    }

    dfs = {}
    dfs['coll'] = get_sheet(ids['coll'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ")
    dfs['gen'] = get_sheet(ids['gen'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ")
    dfs['car_exp'] = get_sheet(ids['car_exp'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ")
    dfs['orders'] = get_sheet(ids['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ")
    dfs['cars'] = get_sheet(ids['cars'], "'قاعدة البيانات'!A:ZZ")
    dfs['clients'] = get_sheet(ids['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ")

    return dfs

# --- 4. PROCESSING LOGIC ---
dfs = load_data()

if dfs:
    # A. Clean Financials
    for key in ['coll', 'gen', 'car_exp']:
        cols = [c for c in dfs[key].columns if 'قيمة' in c]
        for c in cols: dfs[key][c] = dfs[key][c].apply(clean_money)

    # B. Clean Orders
    df_ord = dfs['orders']
    if not df_ord.empty:
        # Dates
        df_ord['Start_Date'] = pd.to_datetime(df_ord['بداية التعاقد'], errors='coerce') # Adjust col name if needed, usually constructed from D/M/Y
        # Cost
        cost_col = next((c for c in df_ord.columns if 'إجمال' in c or 'Total' in c), None)
        if cost_col: df_ord['Total_Cost'] = df_ord[cost_col].apply(clean_money)
        # Dates construction from columns if needed (assuming standard Y/M/D cols exist)
        if 'سنة' in df_ord.columns and 'شهر' in df_ord.columns:
            df_ord['Date'] = pd.to_datetime(df_ord[['سنة', 'شهر', 'يوم']].rename(columns={'سنة':'year','شهر':'month','يوم':'day'}), errors='coerce')
        elif 'Start_Date' in df_ord.columns:
             df_ord['Date'] = df_ord['Start_Date']

    # C. Clean Cars & Generate Owner Liabilities
    df_cars = dfs['cars']
    owner_payments = []
    
    if not df_cars.empty:
        # Identify Columns
        col_pay_amt = next((c for c in df_cars.columns if 'القيمة المالية' in c), None)
        col_pay_freq = next((c for c in df_cars.columns if 'مدة الدفع' in c), None)
        col_pay_start = next((c for c in df_cars.columns if 'تاريخ اول دفع' in c), None)
        col_contract_end = next((c for c in df_cars.columns if 'نهاية التعاقد' in c), None)
        col_status = next((c for c in df_cars.columns if 'حاله التعاقد' in c), None)
        col_id = next((c for c in df_cars.columns if 'No' in c or 'كود' in c), None)
        
        # Name Construction
        c_type = next((c for c in df_cars.columns if 'النوع' in c or 'Type' in c), '')
        c_model = next((c for c in df_cars.columns if 'الطراز' in c or 'Model' in c), '')
        c_year = next((c for c in df_cars.columns if 'سنة' in c or 'Year' in c), '')
        c_color = next((c for c in df_cars.columns if 'اللون' in c or 'Color' in c), '')
        
        df_cars['Full_Name'] = (
            df_cars[c_type].astype(str) + " " + 
            df_cars[c_model].astype(str) + " (" + 
            df_cars[c_year].astype(str) + ") - " + 
            df_cars[c_color].astype(str)
        )
        if col_id:
             df_cars['Clean_ID'] = df_cars[col_id].astype(str).str.strip()
             car_map = df_cars.set_index('Clean_ID')['Full_Name'].to_dict()

        # LIABILITY GENERATOR
        if col_pay_amt and col_pay_start:
            for _, car in df_cars.iterrows():
                try:
                    # Skip if contract not valid (Simple check)
                    # if str(car[col_status]) == 'منتهي': continue 
                    
                    amount = clean_money(car[col_pay_amt])
                    freq = int(clean_money(car[col_pay_freq])) if clean_money(car[col_pay_freq]) > 0 else 45
                    start_date = pd.to_datetime(car[col_pay_start], errors='coerce')
                    end_date = pd.to_datetime(car[col_contract_end], errors='coerce')
                    
                    if pd.isna(start_date) or amount == 0: continue
                    if pd.isna(end_date): end_date = pd.Timestamp.today() + timedelta(days=365) # Fallback

                    # Generate payments
                    curr_date = start_date
                    while curr_date <= end_date and curr_date <= pd.Timestamp.today() + timedelta(days=365):
                        owner_payments.append({
                            'Date': curr_date,
                            'Amount': amount,
                            'Car_ID': str(car[col_id]).strip(),
                            'Car_Name': car['Full_Name'],
                            'Type': 'Owner Fee'
                        })
                        curr_date += timedelta(days=freq)
                except: continue
    
    df_liabilities = pd.DataFrame(owner_payments)

    # --- 5. DASHBOARD UI ---
    
    # SIDEBAR FILTERS
    st.sidebar.title("🔍 أدوات التحكم")
    
    # Prepare Dates for Filter
    all_dates = []
    if 'Date' in df_ord.columns: all_dates.extend(df_ord['Date'].dropna())
    if not df_liabilities.empty: all_dates.extend(df_liabilities['Date'])
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        years = sorted(list(set([d.year for d in all_dates])), reverse=True)
        
        sel_year = st.sidebar.selectbox("السنة (Year)", years)
        months = list(range(1, 13))
        sel_month = st.sidebar.selectbox("الشهر (Month)", months, index=pd.Timestamp.now().month-1)
        
        # FILTER MASKS
        def filter_by_date(df, date_col='Date'):
            if df.empty or date_col not in df.columns: return df
            return df[(df[date_col].dt.year == sel_year) & (df[date_col].dt.month == sel_month)]

        # Apply Filters
        f_ord = filter_by_date(df_ord)
        f_liab = filter_by_date(df_liabilities)
        
        # Financial Filters (Assuming Y/M cols exist)
        def filter_ym(df):
            if df.empty: return df
            # Try to find Year/Month cols
            y_col = next((c for c in df.columns if 'سنة' in c), None)
            m_col = next((c for c in df.columns if 'شهر' in c), None)
            if y_col and m_col:
                return df[(df[y_col].astype(str).str.contains(str(sel_year))) & 
                          (df[m_col].astype(str).str.contains(str(sel_month)))]
            return df

        f_coll = filter_ym(dfs['coll'])
        f_gen = filter_ym(dfs['gen'])
        f_car_exp = filter_ym(dfs['car_exp'])

    else:
        st.error("No Date Data Found")
        st.stop()

    # --- CALCULATIONS ---
    # Revenue
    rev = f_coll['قيمة التحصيل'].sum() if not f_coll.empty else 0
    
    # Expenses
    exp_ops = f_gen['قيمة المصروف'].sum() if not f_gen.empty else 0
    exp_maint = f_car_exp['قيمة المصروف'].sum() if not f_car_exp.empty else 0
    exp_owner = f_liab['Amount'].sum() if not f_liab.empty else 0
    
    total_exp = exp_ops + exp_maint + exp_owner
    net_profit = rev - total_exp

    # HEADER
    st.title(f"📊 تقرير شهر {sel_month} / {sel_year}")
    
    # KPIS
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("صافي الربح (Net)", f"{net_profit:,.0f}", delta="After Owner Fees")
    c2.metric("الإيرادات (Revenue)", f"{rev:,.0f}")
    c3.metric("مستحقات الملاك (Owners)", f"{exp_owner:,.0f}", delta_color="inverse")
    c4.metric("مصروفات الصيانة (Maint)", f"{exp_maint:,.0f}", delta_color="inverse")
    c5.metric("مصروفات إدارية (Ops)", f"{exp_ops:,.0f}", delta_color="inverse")

    st.markdown("---")

    # TABS
    tab_fleet, tab_owners, tab_trends, tab_ai = st.tabs(["🚗 أداء الأسطول", "🤝 مستحقات الملاك", "📈 المؤشرات", "🧠 المدير الذكي"])

    with tab_fleet:
        col_a, col_b = st.columns([2,1])
        with col_a:
            st.subheader("تفاصيل أداء السيارات (Revenue & Profit)")
            
            # Group Revenue by Car
            if not f_ord.empty:
                car_code = next((c for c in f_ord.columns if 'كود السيارة' in c), 'Car_ID')
                car_rev = f_ord.groupby(car_code)['Total_Cost'].sum().reset_index()
                car_rev.columns = ['Car_ID', 'Revenue']
                car_rev['Car_ID'] = car_rev['Car_ID'].astype(str).str.strip()
            else: car_rev = pd.DataFrame(columns=['Car_ID', 'Revenue'])

            # Group Owner Fees
            if not f_liab.empty:
                car_liab = f_liab.groupby('Car_ID')['Amount'].sum().reset_index()
                car_liab.columns = ['Car_ID', 'Owner_Fee']
            else: car_liab = pd.DataFrame(columns=['Car_ID', 'Owner_Fee'])

            # Merge
            stats = pd.merge(car_rev, car_liab, on='Car_ID', how='outer').fillna(0)
            stats['Car_Name'] = stats['Car_ID'].map(car_map).fillna(stats['Car_ID'])
            stats['Net_Profit'] = stats['Revenue'] - stats['Owner_Fee']
            
            # Filter for active/rented only? No, show all that incurred cost or rev
            stats = stats.sort_values('Net_Profit', ascending=False)
            
            st.dataframe(
                stats[['Car_Name', 'Revenue', 'Owner_Fee', 'Net_Profit']],
                use_container_width=True,
                column_config={
                    "Revenue": st.column_config.NumberColumn(format="%d EGP"),
                    "Owner_Fee": st.column_config.NumberColumn(format="%d EGP"),
                    "Net_Profit": st.column_config.ProgressColumn(format="%d EGP", min_value=-5000, max_value=int(stats['Net_Profit'].max() if not stats.empty else 10000))
                }
            )
            
            # Idle Cars Logic
            all_cars = set(car_map.keys())
            active_cars = set(stats['Car_ID'].unique())
            idle_cars = all_cars - active_cars
            if idle_cars:
                st.warning(f"⚠️ السيارات غير المؤجرة هذا الشهر ({len(idle_cars)}): " + ", ".join([car_map.get(c,c) for c in list(idle_cars)[:5]]) + "...")

        with col_b:
            st.subheader("أهم العملاء")
            if not f_ord.empty:
                 client_col = next((c for c in f_ord.columns if 'كود العميل' in c), None)
                 if client_col:
                     top_c = f_ord[client_col].value_counts().head(10).reset_index()
                     top_c.columns = ['ID', 'Rentals']
                     # Name map
                     df_cli = dfs['clients']
                     c_id = next((c for c in df_cli.columns if 'No' in c), None)
                     c_name = next((c for c in df_cli.columns if 'اسم' in c), None)
                     if c_id and c_name:
                         df_cli[c_id] = df_cli[c_id].astype(str).str.strip()
                         cmap = df_cli.set_index(c_id)[c_name].to_dict()
                         top_c['Name'] = top_c['ID'].astype(str).map(cmap).fillna(top_c['ID'])
                     else: top_c['Name'] = top_c['ID']
                     
                     st.table(top_c[['Name', 'Rentals']])

    with tab_owners:
        st.subheader("جدول مدفوعات الملاك المستحقة (Estimated)")
        st.info("تم حساب هذه القيم بناءً على العقود (كل 45 يوم أو حسب العقد) للسيارات السارية فقط.")
        if not f_liab.empty:
            st.dataframe(
                f_liab[['Date', 'Car_Name', 'Amount']].sort_values('Date'),
                use_container_width=True
            )
            st.metric("إجمالي المستحق للملاك هذا الشهر", f"{exp_owner:,.0f} EGP")
        else:
            st.write("لا توجد مستحقات دفع للملاك في هذا الشهر.")

    with tab_trends:
        st.subheader("تطور الأداء المالي (Yearly Trend)")
        # Aggregate by month for the whole year
        if 'Date' in df_liabilities.columns:
            trend_liab = df_liabilities[df_liabilities['Date'].dt.year == sel_year].groupby(df_liabilities['Date'].dt.month)['Amount'].sum()
        else: trend_liab = pd.Series()
        
        # We need a proper date column in Orders for this
        if 'Date' in df_ord.columns:
            trend_rev = df_ord[df_ord['Date'].dt.year == sel_year].groupby(df_ord['Date'].dt.month)['Total_Cost'].sum()
        else: trend_rev = pd.Series()

        trend_df = pd.DataFrame({'Revenue': trend_rev, 'Owner_Pay': trend_liab}).fillna(0)
        st.line_chart(trend_df)

    with tab_ai:
        if st.button("تحليل البيانات بواسطة Gemini"):
            if 'GOOGLE_API_KEY' not in st.secrets:
                st.error("Add API Key")
            else:
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                prompt = f"""
                Analyze my car rental business for {sel_month}/{sel_year}:
                - Revenue: {rev}
                - Owner Obligations: {exp_owner}
                - Maintenance: {exp_maint}
                - Net Profit: {net_profit}
                
                Give me a strategic summary in Arabic regarding cash flow and fleet efficiency.
                """
                advisor = Agent(role='Advisor', goal='Strategy', backstory='Expert', llm=llm)
                task = Task(description=prompt, agent=advisor, expected_output="Summary")
                crew = Crew(agents=[advisor], tasks=[task])
                st.markdown(crew.kickoff())
