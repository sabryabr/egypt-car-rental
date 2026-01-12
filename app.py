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
import string
from datetime import timedelta, datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental Command Center", layout="wide", page_icon="🚘")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    
    /* Metrics */
    .stMetric { 
        background-color: #ffffff !important; 
        border-radius: 12px; 
        padding: 15px; 
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #6c757d !important; font-size: 0.9rem; }
    [data-testid="stMetricValue"] { color: #212529 !important; font-weight: 700; font-size: 1.6rem; }
    
    /* Table Headers */
    .stDataFrame { direction: ltr; }
    
    /* Status Badges */
    .badge-high { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-med { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #198754; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPERS ---
def excel_col_to_index(col_str):
    num = 0
    for c in col_str:
        if c.upper() in string.ascii_uppercase:
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1

def clean_money(x):
    if pd.isna(x) or str(x).strip() == '': return 0.0
    s = str(x).replace(',', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0

def format_currency(x):
    return f"{x:,.0f} EGP"

def format_date(d):
    if pd.isnull(d): return "-"
    return d.strftime("%Y-%m-%d")

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
            res = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = res.get('values', [])
            if not vals: return pd.DataFrame()
            
            if len(vals) > header_row:
                raw_headers = vals[header_row]
                # Fix Duplicates
                headers = []
                seen = {}
                for h in raw_headers:
                    h_str = str(h).strip()
                    if h_str in seen:
                        seen[h_str] += 1
                        headers.append(f"{h_str}_{seen[h_str]}")
                    else:
                        seen[h_str] = 0
                        headers.append(h_str)
                
                data = vals[header_row+1:]
                max_len = len(headers)
                clean_data = [row[:max_len] + [None]*(max_len-len(row)) for row in data]
                return pd.DataFrame(clean_data, columns=headers)
            return pd.DataFrame()
        except Exception: return pd.DataFrame()

    ids = {
        'cars': "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg",
        'coll': "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg",
        'gen': "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q",
        'car_exp': "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE",
        'orders': "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A",
        'clients': "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg"
    }

    dfs = {}
    dfs['cars'] = get_sheet(ids['cars'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0) 
    dfs['coll'] = get_sheet(ids['coll'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
    dfs['gen'] = get_sheet(ids['gen'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
    dfs['car_exp'] = get_sheet(ids['car_exp'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)
    dfs['orders'] = get_sheet(ids['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 0)
    dfs['clients'] = get_sheet(ids['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)

    return dfs

# --- 4. PROCESSING LOGIC ---
dfs = load_data()

if dfs:
    # --- TIME FILTER ---
    st.sidebar.title("📅 Time Control")
    sel_year = st.sidebar.selectbox("Year", [2024, 2025, 2026, 2027], index=2) 
    sel_month = st.sidebar.selectbox("Month", range(1, 13), index=0) 

    # --- A. CARS ENGINE ---
    df_cars_raw = dfs['cars']
    cars_clean = []
    
    if not df_cars_raw.empty:
        raw_values = df_cars_raw.values.tolist()
        for row in raw_values:
            def get(col_letter):
                idx = excel_col_to_index(col_letter)
                if idx < len(row): return row[idx]
                return None

            code = str(get('A') or '').strip()
            if not code or code in ['None', 'No.', 'كود']: continue 

            # Raw Data
            c_type = str(get('B') or '').strip()
            c_model = str(get('E') or '').strip()
            c_year = str(get('H') or '').strip()
            c_color = str(get('J') or '').strip()
            c_seats = str(get('O') or '').strip()
            
            c_name = f"{c_type} {c_model} ({c_year}) - {c_color}"
            
            plate_parts = [get('AC'), get('AB'), get('AA'), get('Z'), get('Y'), get('X'), get('W')]
            plate = " ".join([str(p) for p in plate_parts if p]).strip()
            
            status_raw = str(get('BA') or '')
            is_active = any(x in status_raw for x in ['ساري', 'Valid', 'valid', 'Active'])
            
            km_start = clean_money(get('AV'))
            
            # Dates
            lic_end = pd.to_datetime(get('AQ'), errors='coerce')
            ins_end = pd.to_datetime(get('BK'), errors='coerce')
            contract_end = pd.to_datetime(get('AX'), errors='coerce')
            
            # Payments
            pay_amt = clean_money(get('CJ'))
            pay_freq = clean_money(get('CK'))
            pay_start = pd.to_datetime(get('CL'), errors='coerce')
            
            cars_clean.append({
                'Code': code,
                'Name': c_name,
                'Type': c_type,
                'Model': c_model,
                'Year': c_year,
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

    # --- B. OWNER PAYMENTS (Calculated) ---
    owner_liabilities = []
    total_owner_fees = 0
    
    if not df_cars.empty:
        for _, car in df_cars.iterrows():
            if not car['Active'] or pd.isna(car['Pay_Start']) or car['Pay_Amount'] == 0:
                continue
                
            curr = car['Pay_Start']
            # Fallback end date if missing
            end = car['Contract_End'] if pd.notnull(car['Contract_End']) else datetime(2035, 1, 1)
            
            while curr <= end:
                if curr.year == sel_year and curr.month == sel_month:
                    owner_liabilities.append({
                        'Due Date': curr, # Keep as object for sorting
                        'Car': car['Name'],
                        'Amount': car['Pay_Amount'],
                        'Status': 'Pending' # Simplified for speed
                    })
                    total_owner_fees += car['Pay_Amount']
                curr += timedelta(days=car['Pay_Freq'])

    df_liab = pd.DataFrame(owner_liabilities)
    if not df_liab.empty:
        df_liab = df_liab.sort_values('Due Date') # Sort Sooner to Later
        # Format Date for display
        df_liab['Formatted Date'] = df_liab['Due Date'].dt.strftime('%Y-%m-%d')
        df_liab['Formatted Amount'] = df_liab['Amount'].apply(format_currency)

    # --- C. FINANCIALS (Aggregated) ---
    # Helper for filtering
    def filter_df(df, year, month=None):
        if df.empty: return df
        y_col = next((c for c in df.columns if 'سنة' in c or 'Year' in c), None)
        m_col = next((c for c in df.columns if 'شهر' in c or 'Month' in c), None)
        if y_col and m_col:
            cond = df[y_col].astype(str).str.contains(str(year))
            if month:
                cond = cond & df[m_col].astype(str).str.contains(str(month))
            return df[cond]
        return df

    # Monthly Data
    df_coll_m = filter_df(dfs['coll'], sel_year, sel_month)
    df_gen_m = filter_df(dfs['gen'], sel_year, sel_month)
    df_car_exp_m = filter_df(dfs['car_exp'], sel_year, sel_month)
    
    # Clean Values
    for d in [df_coll_m, df_gen_m, df_car_exp_m]:
        for c in d.columns:
            if 'قيمة' in c: d[c] = d[c].apply(clean_money)

    rev = df_coll_m['قيمة التحصيل'].sum() if not df_coll_m.empty else 0
    exp_ops = df_gen_m['قيمة المصروف'].sum() if not df_gen_m.empty else 0
    exp_maint = df_car_exp_m['قيمة المصروف'].sum() if not df_car_exp_m.empty else 0
    
    total_exp = exp_ops + exp_maint + total_owner_fees
    net_profit = rev - total_exp
    margin = (net_profit / rev * 100) if rev > 0 else 0

    # --- D. ANNUAL TRENDS (Hidden Calculation) ---
    trend_data = []
    # Loop through all months of selected year for trend chart
    df_coll_y = filter_df(dfs['coll'], sel_year)
    df_gen_y = filter_df(dfs['gen'], sel_year)
    df_car_y = filter_df(dfs['car_exp'], sel_year)
    
    # We need to group by month. Assuming 'شهر' column exists and is numeric-ish
    # This is a bit complex with dirty data, so we'll do a robust groupby
    try:
        if not df_coll_y.empty:
            m_col = next((c for c in df_coll_y.columns if 'شهر' in c), None)
            val_col = next((c for c in df_coll_y.columns if 'قيمة' in c), None)
            if m_col and val_col:
                rev_trend = df_coll_y.groupby(m_col)[val_col].sum().reset_index()
                rev_trend.columns = ['Month', 'Revenue']
            else: rev_trend = pd.DataFrame()
    except: rev_trend = pd.DataFrame()

    # --- UI START ---
    st.title(f"📊 Report: {sel_month} / {sel_year}")

    # 1. TOP METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Profit (Final)", format_currency(net_profit), delta=f"{margin:.1f}% Margin")
    m2.metric("Total Revenue", format_currency(rev))
    m3.metric("Owner Liabilities", format_currency(total_owner_fees), delta_color="off")
    m4.metric("Operational Exp", format_currency(exp_ops + exp_maint), delta_color="inverse")

    st.markdown("---")

    # 2. TABS
    tabs = st.tabs(["🚗 Fleet Intelligence", "💰 Financial Analysis", "🤝 Owner Payments", "⚠️ Risk & Alerts", "🧠 AI Insight"])

    # --- TAB 1: FLEET ---
    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        
        # Segregation Logic
        active_cars = df_cars[df_cars['Active'] == True]
        inactive_cars = df_cars[df_cars['Active'] == False]
        
        with c1:
            st.subheader("Fleet Composition")
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Fleet", len(df_cars))
            k2.metric("Active Cars", len(active_cars))
            k3.metric("Inactive/Returned", len(inactive_cars))
            
            if not active_cars.empty:
                # Grouping
                grp_type = active_cars['Type'].value_counts().reset_index()
                grp_type.columns = ['Type', 'Count']
                
                fig_type = px.pie(grp_type, names='Type', values='Count', title="Active Fleet by Type", hole=0.4)
                st.plotly_chart(fig_type, use_container_width=True)

                st.subheader("Active Fleet List (Sorted by Latest Contract)")
                active_cars = active_cars.sort_values('Contract_End', ascending=False)
                
                # Formatted Display
                disp_active = active_cars[['Code', 'Name', 'Plate', 'KM_Start', 'Contract_End']].copy()
                disp_active['Contract_End'] = disp_active['Contract_End'].apply(format_date)
                st.dataframe(disp_active, use_container_width=True)
        
        with c2:
            st.subheader("Inactive Fleet")
            if not inactive_cars.empty:
                inactive_cars = inactive_cars.sort_values('Contract_End', ascending=False)
                disp_inactive = inactive_cars[['Code', 'Name', 'Contract_End']].copy()
                disp_inactive['Contract_End'] = disp_inactive['Contract_End'].apply(format_date)
                st.dataframe(disp_inactive, use_container_width=True)
            else:
                st.info("No inactive cars.")

    # --- TAB 2: FINANCIALS ---
    with tabs[1]:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader(f"Financial Position ({sel_year} Trend)")
            if 'rev_trend' in locals() and not rev_trend.empty:
                fig_trend = px.bar(rev_trend, x='Month', y='Revenue', title="Monthly Revenue Trend")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Trend data requires more entries in Collections sheet.")
                
            # Profit Waterfall
            fig_waterfall = go.Figure(go.Waterfall(
                measure = ["relative", "relative", "relative", "relative", "total"],
                x = ["Revenue", "Owner Fees", "Car Maint", "Ops Expenses", "Net Profit"],
                y = [rev, -total_owner_fees, -exp_maint, -exp_ops, net_profit],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            fig_waterfall.update_layout(title = "Profit Waterfall (This Month)")
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with c2:
            st.subheader("Profit Margin")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = margin,
                title = {'text': "Margin %"},
                gauge = {'axis': {'range': [-10, 100]}, 'bar': {'color': "green" if margin > 20 else "orange"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 3: OWNER PAYMENTS ---
    with tabs[2]:
        st.subheader(f"Owner Obligations: {sel_month}/{sel_year}")
        if not df_liab.empty:
            st.dataframe(
                df_liab[['Formatted Date', 'Car', 'Formatted Amount', 'Status']],
                use_container_width=True
            )
            st.metric("Total Payable", format_currency(total_owner_fees))
        else:
            st.success("No payments due this month.")

    # --- TAB 4: ALERTS (Redesigned) ---
    with tabs[3]:
        st.subheader("🚨 Expiry Risk Management")
        
        today = datetime.today()
        range_high = today + timedelta(days=90)  # 3 Mo
        range_med = today + timedelta(days=180) # 6 Mo
        range_low = today + timedelta(days=365) # 1 Year
        
        alerts = []
        
        for _, car in df_cars.iterrows():
            if not car['Active']: continue
            
            # Check License
            if pd.notnull(car['License_End']):
                risk = None
                if today < car['License_End'] <= range_high: risk = "HIGH (0-3 Mo)"
                elif range_high < car['License_End'] <= range_med: risk = "MEDIUM (3-6 Mo)"
                elif range_med < car['License_End'] <= range_low: risk = "LOW (6-12 Mo)"
                
                if risk:
                    alerts.append({'Car': car['Name'], 'Type': 'License', 'Date': car['License_End'], 'Risk': risk})

            # Check Insurance
            if pd.notnull(car['Insurance_End']):
                risk = None
                if today < car['Insurance_End'] <= range_high: risk = "HIGH (0-3 Mo)"
                elif range_high < car['Insurance_End'] <= range_med: risk = "MEDIUM (3-6 Mo)"
                elif range_med < car['Insurance_End'] <= range_low: risk = "LOW (6-12 Mo)"
                
                if risk:
                    alerts.append({'Car': car['Name'], 'Type': 'Insurance', 'Date': car['Insurance_End'], 'Risk': risk})

        if alerts:
            df_alerts = pd.DataFrame(alerts)
            df_alerts['Date'] = df_alerts['Date'].apply(format_date)
            
            # Sort by Date
            df_alerts = df_alerts.sort_values('Date')
            
            # Separate Tables
            high = df_alerts[df_alerts['Risk'].str.contains("HIGH")]
            med = df_alerts[df_alerts['Risk'].str.contains("MEDIUM")]
            low = df_alerts[df_alerts['Risk'].str.contains("LOW")]
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.error(f"🔴 High Risk ({len(high)})")
                st.dataframe(high[['Car', 'Type', 'Date']], hide_index=True)
            with c2:
                st.warning(f"🟡 Medium Risk ({len(med)})")
                st.dataframe(med[['Car', 'Type', 'Date']], hide_index=True)
            with c3:
                st.success(f"🟢 Low Risk ({len(low)})")
                st.dataframe(low[['Car', 'Type', 'Date']], hide_index=True)
        else:
            st.success("Everything looks good! No upcoming expiries in 12 months.")

    # --- TAB 5: AI ---
    with tabs[4]:
        if st.button("Generate Executive Briefing"):
            if 'GOOGLE_API_KEY' not in st.secrets:
                st.error("Missing API Key")
            else:
                with st.spinner("AI is analyzing all tabs..."):
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                    
                    prompt = f"""
                    Act as the CEO's Strategy Advisor.
                    Report for: {sel_month}/{sel_year}
                    
                    Financials:
                    - Revenue: {rev}
                    - Net Profit: {net_profit} (Margin: {margin:.1f}%)
                    - Owner Obligations: {total_owner_fees}
                    
                    Fleet:
                    - Total Cars: {len(df_cars)}
                    - Active: {len(active_cars)}
                    
                    Risks:
                    - High Risk Expiries: {len(high) if 'high' in locals() else 0}
                    
                    Write a concise, bulleted Executive Summary in Arabic focusing on:
                    1. Profitability Health.
                    2. Cash Flow warnings (Owner payments).
                    3. Critical Action Items (Renewals).
                    """
                    advisor = Agent(role='Advisor', goal='Report', backstory='Expert', llm=llm)
                    task = Task(description=prompt, agent=advisor, expected_output="Summary")
                    crew = Crew(agents=[advisor], tasks=[task])
                    st.markdown(crew.kickoff())
