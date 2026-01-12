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
    
    /* Metrics */
    .stMetric { 
        background-color: #ffffff !important; 
        border-radius: 15px; 
        padding: 20px; 
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; font-size: 1.1rem; font-weight: bold; }
    [data-testid="stMetricValue"] { color: #2c3e50 !important; font-weight: 900; font-size: 2.2rem; }
    
    /* Tables */
    .stDataFrame { direction: ltr; }
    
    /* Risk Badges */
    .risk-high { background-color: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .risk-med { background-color: #fff3e0; color: #ef6c00; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .risk-low { background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
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
            c_color = str(get('I') or '').strip()
            
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
        # Find critical columns
        col_start = next((c for c in df_orders.columns if 'بداية' in c or 'Start' in c), None)
        col_end = next((c for c in df_orders.columns if 'نهاية' in c or 'End' in c), None)
        col_car = next((c for c in df_orders.columns if 'كود السيارة' in c or 'Car' in c), None)
        col_cost = next((c for c in df_orders.columns if 'إجمال' in c or 'Total' in c), None)
        
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
    # Selected Month Logic
    start_m = datetime(sel_year, sel_month, 1)
    _, last_day = calendar.monthrange(sel_year, sel_month)
    end_m = datetime(sel_year, sel_month, last_day, 23, 59, 59)
    days_in_month = last_day
    
    # Calculate Occupied Days for Each Car
    occupancy_data = []
    
    for _, car in df_cars.iterrows():
        if not car['Active']: continue
        
        rented_days = 0
        if not df_ord_clean.empty:
            car_orders = df_ord_clean[df_ord_clean['Car_Code'] == car['Code']]
            for _, o in car_orders.iterrows():
                # Overlap Logic
                latest_start = max(start_m, o['Start'])
                earliest_end = min(end_m, o['End'])
                delta = (earliest_end - latest_start).days + 1
                if delta > 0:
                    rented_days += delta
        
        # Cap at days_in_month (in case of overlaps errors)
        rented_days = min(rented_days, days_in_month)
        rate = (rented_days / days_in_month) * 100
        
        occupancy_data.append({
            'Car': car['Full_Name'],
            'Type': car['Type'],
            'Model': car['Model'],
            'Year': car['Year'],
            'Rented_Days': rented_days,
            'Idle_Days': days_in_month - rented_days,
            'Rate': rate
        })
    
    df_occupancy = pd.DataFrame(occupancy_data)
    
    # Fleet Totals
    total_active = len(df_cars[df_cars['Active']])
    avg_occupancy = df_occupancy['Rate'].mean() if not df_occupancy.empty else 0
    total_rented_days = df_occupancy['Rented_Days'].sum() if not df_occupancy.empty else 0
    fleet_capacity_days = total_active * days_in_month
    fleet_utilization = (total_rented_days / fleet_capacity_days * 100) if fleet_capacity_days > 0 else 0

    # --- D. OWNER LIABILITIES ---
    owner_liabilities = []
    total_owner_fees = 0
    
    if not df_cars.empty:
        for _, car in df_cars.iterrows():
            if not car['Active'] or pd.isna(car['Pay_Start']) or car['Owner_Fee'] == 0:
                continue
            curr = car['Pay_Start']
            end = car['Contract_End'] if pd.notnull(car['Contract_End']) else datetime(2035, 1, 1)
            while curr <= end:
                if curr.year == sel_year and curr.month == sel_month:
                    owner_liabilities.append({
                        'Date': curr, 'Car': car['Full_Name'], 'Amount': car['Owner_Fee'], 'Status': 'Scheduled'
                    })
                    total_owner_fees += car['Owner_Fee']
                curr += timedelta(days=car['Pay_Freq'])
    df_liab = pd.DataFrame(owner_liabilities)
    if not df_liab.empty:
        df_liab = df_liab.sort_values('Date')
        df_liab['Date'] = df_liab['Date'].apply(format_date)
        df_liab['Amount'] = df_liab['Amount'].apply(format_egp)

    # --- E. FINANCIALS ---
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

    df_coll_m = filter_df(dfs['coll'], sel_year, sel_month)
    df_gen_m = filter_df(dfs['gen'], sel_year, sel_month)
    df_car_exp_m = filter_df(dfs['car_exp'], sel_year, sel_month)
    
    for d in [df_coll_m, df_gen_m, df_car_exp_m]:
        for c in d.columns:
            if 'قيمة' in c: d[c] = d[c].apply(clean_money)

    rev = df_coll_m['قيمة التحصيل'].sum() if not df_coll_m.empty else 0
    exp_ops = df_gen_m['قيمة المصروف'].sum() if not df_gen_m.empty else 0
    exp_maint = df_car_exp_m['قيمة المصروف'].sum() if not df_car_exp_m.empty else 0
    total_exp = exp_ops + exp_maint + total_owner_fees
    net_profit = rev - total_exp
    margin = (net_profit / rev * 100) if rev > 0 else 0

    # --- UI START ---
    st.title(f"📊 Executive Dashboard: {sel_month} / {sel_year}")

    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Profit", format_egp(net_profit), delta=f"{margin:.1f}% Margin")
    m2.metric("Revenue", format_egp(rev))
    m3.metric("Fleet Occupancy", f"{fleet_utilization:.1f}%", delta="Utilization")
    m4.metric("Owner Fees", format_egp(total_owner_fees), delta_color="off")

    st.divider()

    # TABS
    tabs = st.tabs(["📅 Utilization & Calendar", "🚗 Fleet 360°", "💰 Financials", "⚠️ Risk Management", "🤝 Owner Payments", "📋 All Cars", "🧠 AI"])

    # --- TAB 1: CALENDAR & OCCUPANCY ---
    with tabs[0]:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("📊 Occupancy by Car Type")
            if not df_occupancy.empty:
                # Aggregated by Type
                occ_type = df_occupancy.groupby('Type')['Rented_Days'].sum() / (df_occupancy.groupby('Type')['Car'].count() * days_in_month) * 100
                occ_type = occ_type.reset_index(name='Occupancy %')
                fig_occ = px.bar(occ_type, x='Type', y='Occupancy %', color='Occupancy %', color_continuous_scale='Bluered')
                fig_occ.update_layout(height=400, font=dict(size=14, family="Arial Black"))
                st.plotly_chart(fig_occ, use_container_width=True)
            else: st.info("No data")
            
        with c2:
            st.subheader("🏆 Top Performing Cars (This Month)")
            if not df_occupancy.empty:
                top_cars = df_occupancy.sort_values('Rate', ascending=False).head(10)
                fig_top = px.bar(top_cars, x='Rate', y='Car', orientation='h', title="Highest Occupancy %")
                fig_top.update_layout(height=400, font=dict(size=14, family="Arial Black"), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_top, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🗓️ Fleet Schedule (Gantt Chart)")
        
        # Filter Orders for Timeline (Only this month)
        if not df_ord_clean.empty:
            timeline_orders = df_ord_clean[
                (df_ord_clean['Start'] <= end_m) & (df_ord_clean['End'] >= start_m)
            ].copy()
            
            if not timeline_orders.empty:
                # Add Car Names
                car_map = df_cars.set_index('Code')['Full_Name'].to_dict()
                timeline_orders['Car Name'] = timeline_orders['Car_Code'].map(car_map).fillna(timeline_orders['Car_Code'])
                
                fig_gantt = px.timeline(timeline_orders, x_start="Start", x_end="End", y="Car Name", color="Cost", title="Rental Schedule")
                fig_gantt.update_yaxes(autorange="reversed") # Cars top to bottom
                fig_gantt.update_layout(height=600, font=dict(size=14), xaxis_range=[start_m, end_m])
                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.info("No orders found for this month.")
        
        st.markdown("---")
        st.subheader("📈 Daily Fleet Pulse (Active vs Idle)")
        
        # Calculate Daily Status
        daily_stats = []
        for d in range(1, days_in_month + 1):
            day_date = datetime(sel_year, sel_month, d)
            rented_count = 0
            if not df_ord_clean.empty:
                active_ords = df_ord_clean[(df_ord_clean['Start'] <= day_date) & (df_ord_clean['End'] >= day_date)]
                rented_count = len(active_ords)
            
            daily_stats.append({
                'Day': day_date,
                'Rented': rented_count,
                'Idle': total_active - rented_count
            })
        
        df_daily = pd.DataFrame(daily_stats)
        fig_pulse = go.Figure()
        fig_pulse.add_trace(go.Scatter(x=df_daily['Day'], y=df_daily['Rented'], mode='lines', name='Rented Cars', stackgroup='one', line=dict(color='green')))
        fig_pulse.add_trace(go.Scatter(x=df_daily['Day'], y=df_daily['Idle'], mode='lines', name='Idle Cars', stackgroup='one', line=dict(color='lightgray')))
        fig_pulse.update_layout(title="Daily Fleet Status (Rented vs Idle)", height=450, font=dict(size=14))
        st.plotly_chart(fig_pulse, use_container_width=True)

    # --- TAB 2: FLEET ---
    with tabs[1]:
        c1, c2 = st.columns([2, 1])
        active_cars = df_cars[df_cars['Active'] == True]
        
        with c1:
            st.subheader("Fleet Segregation")
            if not active_cars.empty:
                fig_sun = px.sunburst(active_cars, path=['Type', 'Model'], title="Breakdown by Type > Model")
                fig_sun.update_layout(height=500, font=dict(size=14))
                st.plotly_chart(fig_sun, use_container_width=True)

            st.subheader("Active Fleet List")
            if not active_cars.empty:
                active_sorted = active_cars.sort_values('Contract_End', ascending=True)
                disp = active_sorted[['Code', 'Full_Name', 'Plate', 'KM', 'Contract_End']].copy()
                disp['Contract_End'] = disp['Contract_End'].apply(format_date)
                st.dataframe(disp, use_container_width=True, hide_index=True)

        with c2:
            st.subheader("By Year Model")
            if not active_cars.empty:
                grp_year = active_cars['Year'].value_counts().reset_index()
                grp_year.columns = ['Year', 'Count']
                fig_year = px.bar(grp_year, x='Year', y='Count')
                fig_year.update_layout(height=500, font=dict(size=14))
                st.plotly_chart(fig_year, use_container_width=True)
            
            st.metric("Total Active Fleet", len(active_cars))

    # --- TAB 3: FINANCIALS ---
    with tabs[2]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Monthly Trend")
            # Trend calculation
            trend_data = []
            for m in range(1, 13):
                r = filter_df(dfs['coll'], sel_year, m)['قيمة التحصيل'].apply(clean_money).sum()
                g = filter_df(dfs['gen'], sel_year, m)['قيمة المصروف'].apply(clean_money).sum()
                c = filter_df(dfs['car_exp'], sel_year, m)['قيمة المصروف'].apply(clean_money).sum()
                trend_data.append({'Month': m, 'Revenue': r, 'Expenses': g+c})
            
            df_trend = pd.DataFrame(trend_data)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend['Month'], y=df_trend['Revenue'], name='Revenue', line=dict(color='green', width=4)))
            fig_trend.add_trace(go.Scatter(x=df_trend['Month'], y=df_trend['Expenses'], name='Expenses', line=dict(color='red', width=4)))
            fig_trend.update_layout(height=500, font=dict(size=14))
            st.plotly_chart(fig_trend, use_container_width=True)

        with c2:
            st.subheader("Margin Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = margin, title = {'text': "Margin %"},
                gauge = {'axis': {'range': [-20, 100]}, 'bar': {'color': "green" if margin > 15 else "orange"}}
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 4: RISKS ---
    with tabs[3]:
        st.subheader("🚨 Expiry Risk Management")
        today = datetime.today()
        limits = {
            'High (0-3 Mo)': today + timedelta(days=90),
            'Med (3-6 Mo)': today + timedelta(days=180),
            'Low (6-12 Mo)': today + timedelta(days=365)
        }
        
        def get_risk(d):
            if pd.isnull(d): return None
            if today < d <= limits['High (0-3 Mo)']: return "High"
            if limits['High (0-3 Mo)'] < d <= limits['Med (3-6 Mo)']: return "Med"
            if limits['Med (3-6 Mo)'] < d <= limits['Low (6-12 Mo)']: return "Low"
            return None

        risks = []
        for _, car in df_cars.iterrows():
            if not car['Active']: continue
            r_lic = get_risk(car['License'])
            if r_lic: risks.append({'Car': car['Full_Name'], 'Type': 'License', 'Date': car['License'], 'Risk': r_lic})
            r_ins = get_risk(car['Insurance'])
            if r_ins: risks.append({'Car': car['Full_Name'], 'Type': 'Insurance', 'Date': car['Insurance'], 'Risk': r_ins})

        if risks:
            df_risk = pd.DataFrame(risks)
            df_risk['Date'] = df_risk['Date'].apply(format_date)
            df_risk = df_risk.sort_values('Date')
            
            c1, c2, c3 = st.columns(3)
            with c1:
                high = df_risk[df_risk['Risk'] == 'High']
                st.markdown(f"<div class='risk-high'>🔴 High Risk ({len(high)})</div>", unsafe_allow_html=True)
                st.dataframe(high[['Car', 'Type', 'Date']], hide_index=True)
            with c2:
                med = df_risk[df_risk['Risk'] == 'Med']
                st.markdown(f"<div class='risk-med'>🟠 Medium Risk ({len(med)})</div>", unsafe_allow_html=True)
                st.dataframe(med[['Car', 'Type', 'Date']], hide_index=True)
            with c3:
                low = df_risk[df_risk['Risk'] == 'Low']
                st.markdown(f"<div class='risk-low'>🟢 Low Risk ({len(low)})</div>", unsafe_allow_html=True)
                st.dataframe(low[['Car', 'Type', 'Date']], hide_index=True)
        else:
            st.success("✅ No risks found.")

    # --- TAB 5: PAYMENTS ---
    with tabs[4]:
        st.subheader("Owner Payment Schedule")
        if not df_liab.empty:
            st.dataframe(df_liab[['Date', 'Car', 'Amount', 'Status']], use_container_width=True, hide_index=True)
        else:
            st.info("No payments due.")

    # --- TAB 6: ALL CARS ---
    with tabs[5]:
        st.subheader("📋 Full Database")
        disp_full = df_cars.copy()
        for c in ['License', 'Insurance', 'Contract_End', 'Pay_Start']:
            disp_full[c] = disp_full[c].apply(format_date)
        disp_full['Owner_Fee'] = disp_full['Owner_Fee'].apply(format_egp)
        st.dataframe(disp_full, use_container_width=True, hide_index=True)

    # --- TAB 7: AI ---
    with tabs[6]:
        if st.button("Generate Strategy"):
            if 'GOOGLE_API_KEY' not in st.secrets:
                st.error("Missing API Key")
            else:
                with st.spinner("Analyzing..."):
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                    prompt = f"Role: CEO Advisor. Data: {sel_month}/{sel_year}. Rev: {rev}, Profit: {net_profit}, Fleet Util: {fleet_utilization:.1f}%. Task: Arabic Briefing."
                    advisor = Agent(role='Advisor', goal='Report', backstory='Expert', llm=llm)
                    task = Task(description=prompt, agent=advisor, expected_output="Summary")
                    crew = Crew(agents=[advisor], tasks=[task])
                    st.markdown(crew.kickoff())
