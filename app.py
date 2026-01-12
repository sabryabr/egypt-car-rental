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
import calendar

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental Command Center", layout="wide", page_icon="🚘")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    
    /* Bigger, Bolder Metrics */
    .stMetric { 
        background-color: #ffffff !important; 
        border-radius: 15px; 
        padding: 20px; 
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    [data-testid="stMetricLabel"] { color: #555 !important; font-size: 1.2rem; font-weight: 800; }
    [data-testid="stMetricValue"] { color: #2c3e50 !important; font-weight: 900; font-size: 2.5rem; }
    
    /* Clean Tables */
    .stDataFrame { direction: ltr; }
    
    /* Risk Badges */
    .risk-high { background-color: #ffebee; color: #c62828; padding: 6px 10px; border-radius: 8px; font-weight: bold; border: 1px solid #c62828; }
    .risk-med { background-color: #fff3e0; color: #ef6c00; padding: 6px 10px; border-radius: 8px; font-weight: bold; border: 1px solid #ef6c00; }
    .risk-low { background-color: #e8f5e9; color: #2e7d32; padding: 6px 10px; border-radius: 8px; font-weight: bold; border: 1px solid #2e7d32; }
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

def format_egp(x):
    return f"{x:,.0f} EGP"

def format_date(d):
    if pd.isnull(d): return "-"
    return d.strftime("%Y-%m-%d")

# Safe column finder
def get_col(df, keyword):
    for c in df.columns:
        if keyword in str(c): return c
    return None

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
    st.sidebar.header("🗓️ Time Machine")
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

            c_type = str(get('B') or '').strip()
            c_model = str(get('E') or '').strip()
            c_year = str(get('H') or '').strip()
            c_color = str(get('I') or '').strip() # Fixed Color Column
            
            c_name = f"{c_type} {c_model} ({c_year}) - {c_color}"
            
            plate_parts = [get('AC'), get('AB'), get('AA'), get('Z'), get('Y'), get('X'), get('W')]
            plate = " ".join([str(p) for p in plate_parts if p]).strip()
            
            status_raw = str(get('BA') or '')
            is_active = any(x in status_raw for x in ['ساري', 'Valid', 'valid', 'Active'])
            
            km_start = clean_money(get('AV'))
            
            lic_end = pd.to_datetime(get('AQ'), errors='coerce')
            ins_end = pd.to_datetime(get('BK'), errors='coerce')
            contract_end = pd.to_datetime(get('AX'), errors='coerce')
            
            pay_amt = clean_money(get('CJ'))
            pay_freq = clean_money(get('CK'))
            pay_start = pd.to_datetime(get('CL'), errors='coerce')
            
            cars_clean.append({
                'Code': code,
                'Full_Name': c_name,
                'Type': c_type,
                'Model': c_model,
                'Year': c_year,
                'Plate': plate,
                'Active': is_active,
                'KM': km_start,
                'License': lic_end,
                'Insurance': ins_end,
                'Contract_End': contract_end,
                'Owner_Fee': pay_amt,
                'Pay_Freq': int(pay_freq) if pay_freq > 0 else 45,
                'Pay_Start': pay_start
            })
    
    df_cars = pd.DataFrame(cars_clean)
    
    # --- B. ORDERS ENGINE ---
    df_orders = dfs['orders']
    orders_clean = []
    if not df_orders.empty:
        # Robust column finding
        col_start = get_col(df_orders, 'بداية') or get_col(df_orders, 'Start')
        col_end = get_col(df_orders, 'نهاية') or get_col(df_orders, 'End')
        col_car = get_col(df_orders, 'كود السيارة') or get_col(df_orders, 'Car')
        col_cost = get_col(df_orders, 'إجمال') or get_col(df_orders, 'Total')
        
        if col_start and col_car:
            for _, row in df_orders.iterrows():
                try:
                    s_date = pd.to_datetime(row[col_start], errors='coerce')
                    e_date = pd.to_datetime(row[col_end], errors='coerce') if col_end else s_date
                    if pd.isna(s_date): continue
                    if pd.isna(e_date): e_date = s_date + timedelta(days=1)
                    
                    orders_clean.append({
                        'Start': s_date,
                        'End': e_date,
                        'Car_Code': str(row[col_car]).strip(),
                        'Cost': clean_money(row[col_cost]) if col_cost else 0
                    })
                except: continue
    
    df_ord_clean = pd.DataFrame(orders_clean)

    # --- C. UTILIZATION & OCCUPANCY ---
    start_m = datetime(sel_year, sel_month, 1)
    _, last_day = calendar.monthrange(sel_year, sel_month)
    end_m = datetime(sel_year, sel_month, last_day, 23, 59, 59)
    days_in_month = last_day
    
    occupancy_data = []
    
    for _, car in df_cars.iterrows():
        if not car['Active']: continue
        
        rented_days = 0
        if not df_ord_clean.empty:
            car_orders = df_ord_clean[df_ord_clean['Car_Code'] == car['Code']]
            for _, o in car_orders.iterrows():
                latest_start = max(start_m, o['Start'])
                earliest_end = min(end_m, o['End'])
                delta = (earliest_end - latest_start).days + 1
                if delta > 0:
                    rented_days += delta
        
        rented_days = min(rented_days, days_in_month)
        rate = (rented_days / days_in_month) * 100
        
        occupancy_data.append({
            'Car': car['Full_Name'],
            'Type': car['Type'],
            'Model': car['Model'],
            'Year': car['Year'],
            'Rented_Days': rented_days,
            'Rate': rate
        })
    
    df_occupancy = pd.DataFrame(occupancy_data)
    
    total_active = len(df_cars[df_cars['Active']])
    total_rented_days = df_occupancy['Rented_Days'].sum() if not df_occupancy.empty else 0
    fleet_capacity_days = total_active * days_in_month
    fleet_utilization = (total_rented_days / fleet_capacity_days * 100) if fleet_capacity_days > 0 else 0

    # --- D. FINANCIALS ---
    def filter_df(df, year, month=None):
        if df.empty: return df
        y_col = get_col(df, 'سنة') or get_col(df, 'Year')
        m_col = get_col(df, 'شهر') or get_col(df, 'Month')
        if y_col and m_col:
            cond = df[y_col].astype(str).str.contains(str(year))
            if month:
                cond = cond & df[m_col].astype(str).str.contains(str(month))
            return df[cond]
        return df

    df_coll_m = filter_df(dfs['coll'], sel_year, sel_month)
    df_gen_m = filter_df(dfs['gen'], sel_year, sel_month)
    df_car_exp_m = filter_df(dfs['car_exp'], sel_year, sel_month)
    
    # Safe Sum using get_col
    def safe_sum(df, keyword):
        col = get_col(df, keyword)
        if col: return df[col].apply(clean_money).sum()
        return 0

    rev = safe_sum(df_coll_m, 'قيمة')
    exp_ops = safe_sum(df_gen_m, 'قيمة')
    exp_maint = safe_sum(df_car_exp_m, 'قيمة')
    
    # Owner Fees Logic
    owner_liabilities = []
    total_owner_fees = 0
    if not df_cars.empty:
        for _, car in df_cars.iterrows():
            if not car['Active'] or pd.isna(car['Pay_Start']) or car['Owner_Fee'] == 0: continue
            curr = car['Pay_Start']
            end = car['Contract_End'] if pd.notnull(car['Contract_End']) else datetime(2035, 1, 1)
            while curr <= end:
                if curr.year == sel_year and curr.month == sel_month:
                    owner_liabilities.append({'Date': curr, 'Car': car['Full_Name'], 'Amount': car['Owner_Fee']})
                    total_owner_fees += car['Owner_Fee']
                curr += timedelta(days=car['Pay_Freq'])
    
    total_exp = exp_ops + exp_maint + total_owner_fees
    net_profit = rev - total_exp
    margin = (net_profit / rev * 100) if rev > 0 else 0

    # --- UI ---
    st.title(f"📊 Executive Dashboard: {sel_month} / {sel_year}")

    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Profit", format_egp(net_profit), delta=f"{margin:.1f}% Margin")
    m2.metric("Revenue", format_egp(rev))
    m3.metric("Fleet Occupancy", f"{fleet_utilization:.1f}%", delta="Utilization")
    m4.metric("Owner Fees", format_egp(total_owner_fees), delta_color="off")

    st.divider()

    tabs = st.tabs(["📅 Utilization & Calendar", "🚗 Fleet 360°", "💰 Financials", "⚠️ Risk Management", "🤝 Owner Payments", "📋 All Cars", "🧠 AI"])

    # TAB 1: CALENDAR
    with tabs[0]:
        st.subheader("🗓️ Monthly Fleet Schedule")
        if not df_ord_clean.empty:
            timeline = df_ord_clean[
                (df_ord_clean['Start'] <= end_m) & (df_ord_clean['End'] >= start_m)
            ].copy()
            if not timeline.empty:
                car_map = df_cars.set_index('Code')['Full_Name'].to_dict()
                timeline['Car'] = timeline['Car_Code'].map(car_map).fillna(timeline['Car_Code'])
                
                fig = px.timeline(timeline, x_start="Start", x_end="End", y="Car", color="Cost", title="Rental Gantt Chart")
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=600, font=dict(size=14, family="Arial Black"), xaxis_range=[start_m, end_m])
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No rentals this month.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Occupancy by Type")
            if not df_occupancy.empty:
                occ_type = df_occupancy.groupby('Type')['Rate'].mean().reset_index()
                fig_occ = px.bar(occ_type, x='Type', y='Rate', color='Rate', color_continuous_scale='Greens')
                fig_occ.update_layout(height=400, font=dict(size=14))
                st.plotly_chart(fig_occ, use_container_width=True)
        
        with c2:
            st.subheader("Top Occupied Cars")
            if not df_occupancy.empty:
                top = df_occupancy.sort_values('Rate', ascending=False).head(10)
                st.dataframe(top[['Car', 'Rented_Days', 'Rate']], hide_index=True, use_container_width=True)

    # TAB 2: FLEET
    with tabs[1]:
        c1, c2 = st.columns([2, 1])
        active_cars = df_cars[df_cars['Active'] == True]
        with c1:
            st.subheader("Fleet Segregation")
            if not active_cars.empty:
                fig = px.sunburst(active_cars, path=['Type', 'Model'], title="Active Fleet Breakdown")
                fig.update_layout(height=500, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Active Fleet (Sorted by Contract)")
                disp = active_cars.sort_values('Contract_End', ascending=True)[['Code', 'Full_Name', 'Plate', 'Contract_End']]
                disp['Contract_End'] = disp['Contract_End'].apply(format_date)
                st.dataframe(disp, hide_index=True, use_container_width=True)
        
        with c2:
            st.subheader("Model Year Distribution")
            if not active_cars.empty:
                fig_y = px.bar(active_cars['Year'].value_counts().reset_index(), x='Year', y='count')
                st.plotly_chart(fig_y, use_container_width=True)
                st.metric("Total Active", len(active_cars))

    # TAB 3: FINANCIALS
    with tabs[2]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Financial Trend")
            trend = []
            for m in range(1, 13):
                r = safe_sum(filter_df(dfs['coll'], sel_year, m), 'قيمة')
                e = safe_sum(filter_df(dfs['gen'], sel_year, m), 'قيمة') + safe_sum(filter_df(dfs['car_exp'], sel_year, m), 'قيمة')
                trend.append({'Month': m, 'Revenue': r, 'Expenses': e})
            df_trend = pd.DataFrame(trend)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trend['Month'], y=df_trend['Revenue'], name='Rev', line=dict(color='green', width=4)))
            fig.add_trace(go.Scatter(x=df_trend['Month'], y=df_trend['Expenses'], name='Exp', line=dict(color='red', width=4)))
            fig.update_layout(height=500, font=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Profit Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=margin, title={'text': "Margin %"},
                gauge={'axis': {'range': [-20, 100]}, 'bar': {'color': "green" if margin > 15 else "orange"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: RISKS
    with tabs[3]:
        st.subheader("🚨 Expiry Risks")
        today = datetime.today()
        limits = {
            'High': today + timedelta(days=90),
            'Med': today + timedelta(days=180),
            'Low': today + timedelta(days=365)
        }
        
        def get_risk(d):
            if pd.isnull(d): return None
            if today < d <= limits['High']: return "High"
            if limits['High'] < d <= limits['Med']: return "Med"
            if limits['Med'] < d <= limits['Low']: return "Low"
            return None

        risks = []
        for _, car in df_cars.iterrows():
            if not car['Active']: continue
            for kind, col in [('License', 'License'), ('Insurance', 'Insurance')]:
                r = get_risk(car[col])
                if r: risks.append({'Car': car['Full_Name'], 'Type': kind, 'Date': format_date(car[col]), 'Risk': r})
        
        if risks:
            df_risk = pd.DataFrame(risks).sort_values('Date')
            c1, c2, c3 = st.columns(3)
            with c1:
                h = df_risk[df_risk['Risk']=='High']
                st.markdown(f"<div class='risk-high'>🔴 High Risk ({len(h)})</div>", unsafe_allow_html=True)
                st.dataframe(h[['Car', 'Type', 'Date']], hide_index=True)
            with c2:
                m = df_risk[df_risk['Risk']=='Med']
                st.markdown(f"<div class='risk-med'>🟠 Medium Risk ({len(m)})</div>", unsafe_allow_html=True)
                st.dataframe(m[['Car', 'Type', 'Date']], hide_index=True)
            with c3:
                l = df_risk[df_risk['Risk']=='Low']
                st.markdown(f"<div class='risk-low'>🟢 Low Risk ({len(l)})</div>", unsafe_allow_html=True)
                st.dataframe(l[['Car', 'Type', 'Date']], hide_index=True)
        else:
            st.success("No upcoming expiries.")

    # TAB 5: PAYMENTS
    with tabs[4]:
        st.subheader("Owner Payment Schedule")
        if owner_liabilities:
            df_pay = pd.DataFrame(owner_liabilities).sort_values('Date')
            df_pay['Date'] = df_pay['Date'].apply(format_date)
            df_pay['Amount'] = df_pay['Amount'].apply(format_egp)
            st.dataframe(df_pay, hide_index=True, use_container_width=True)
        else: st.info("No payments.")

    # TAB 6: ALL CARS
    with tabs[5]:
        st.subheader("📋 Full Fleet List")
        disp = df_cars.copy()
        for c in ['License', 'Insurance', 'Contract_End', 'Pay_Start']:
            disp[c] = disp[c].apply(format_date)
        disp['Owner_Fee'] = disp['Owner_Fee'].apply(format_egp)
        st.dataframe(disp, hide_index=True, use_container_width=True)

    # TAB 7: AI
    with tabs[6]:
        if st.button("Generate Strategy"):
            if 'GOOGLE_API_KEY' not in st.secrets:
                st.error("Missing API Key")
            else:
                with st.spinner("Analyzing..."):
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                    prompt = f"CEO Advisor Report: {sel_month}/{sel_year}. Rev: {rev}, Profit: {net_profit}, Util: {fleet_utilization:.1f}%. Arabic Brief."
                    advisor = Agent(role='Advisor', goal='Report', backstory='Expert', llm=llm)
                    task = Task(description=prompt, agent=advisor, expected_output="Summary")
                    crew = Crew(agents=[advisor], tasks=[task])
                    st.markdown(crew.kickoff())
