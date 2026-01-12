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
    .risk-high { background-color: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 6px; font-weight: bold; border: 1px solid #c62828; }
    .risk-med { background-color: #fff3e0; color: #ef6c00; padding: 4px 8px; border-radius: 6px; font-weight: bold; border: 1px solid #ef6c00; }
    .risk-low { background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 6px; font-weight: bold; border: 1px solid #2e7d32; }
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
            
            # Smart Header Handling
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
    # Orders has headers on Row 1 (Index 1) based on snippet
    dfs['orders'] = get_sheet(ids['orders'], "'صفحة الإدخالات للإيجارات'!A:ZZ", 1)
    dfs['clients'] = get_sheet(ids['clients'], "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", 0)

    return dfs

# --- 4. PROCESSING LOGIC ---
dfs = load_data()

if dfs:
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
            
            plate = f"{get('AC') or ''} {get('AB') or ''} {get('AA') or ''} {get('Z') or ''} {get('Y') or ''} {get('X') or ''} {get('W') or ''}".strip()
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
                'Code': code, 'Full_Name': c_name, 'Type': c_type, 'Model': c_model, 'Year': c_year, 'Color': c_color,
                'Plate': plate, 'Active': is_active, 'KM': km_start, 'License': lic_end, 'Insurance': ins_end,
                'Contract_End': contract_end, 'Owner_Fee': pay_amt, 'Pay_Freq': int(pay_freq) if pay_freq > 0 else 45, 'Pay_Start': pay_start
            })
    
    df_cars = pd.DataFrame(cars_clean)

    # --- B. ORDERS ENGINE (FIXED SPLIT DATES) ---
    df_orders = dfs['orders']
    orders_clean = []
    
    if not df_orders.empty:
        # Columns based on snippet (Start: D, E, F | End: M, N, O approx)
        # We look for "يوم" (Day), "شهر" (Month), "سنة" (Year)
        # Since they are duplicated, we use position logic
        
        # Identify columns
        try:
            # Map columns to list
            cols = list(df_orders.columns)
            # Find indices of 'يوم', 'شهر', 'سنة'
            day_idxs = [i for i, c in enumerate(cols) if 'يوم' in str(c)]
            month_idxs = [i for i, c in enumerate(cols) if 'شهر' in str(c)]
            year_idxs = [i for i, c in enumerate(cols) if 'سنة' in str(c)]
            
            # Start Date is usually the first set
            s_d_idx = day_idxs[0] if day_idxs else None
            s_m_idx = month_idxs[0] if month_idxs else None
            s_y_idx = year_idxs[0] if year_idxs else None
            
            # End Date is the second set (if exists)
            e_d_idx = day_idxs[1] if len(day_idxs) > 1 else None
            e_m_idx = month_idxs[1] if len(month_idxs) > 1 else None
            e_y_idx = year_idxs[1] if len(year_idxs) > 1 else None
            
            car_col_idx = next((i for i, c in enumerate(cols) if 'كود السيارة' in str(c)), 2) # Default index 2
            cost_col_idx = next((i for i, c in enumerate(cols) if 'إجمال' in str(c)), None)

            for i, row in df_orders.iterrows():
                try:
                    # Construct Start Date
                    s_day = int(clean_money(row.iloc[s_d_idx]))
                    s_month = int(clean_money(row.iloc[s_m_idx]))
                    s_year = int(clean_money(row.iloc[s_y_idx]))
                    
                    if s_year < 2000: continue # Invalid
                    start_date = datetime(s_year, s_month, s_day)
                    
                    # Construct End Date
                    if e_d_idx:
                        e_day = int(clean_money(row.iloc[e_d_idx]))
                        e_month = int(clean_money(row.iloc[e_m_idx]))
                        e_year = int(clean_money(row.iloc[e_y_idx]))
                        end_date = datetime(e_year, e_month, e_day)
                    else:
                        end_date = start_date + timedelta(days=1)
                        
                    car_code = str(row.iloc[car_col_idx]).strip()
                    cost = clean_money(row.iloc[cost_col_idx]) if cost_col_idx else 0
                    
                    orders_clean.append({
                        'Start': start_date, 'End': end_date, 'Car_Code': car_code, 'Cost': cost
                    })
                except: continue
        except: pass
    
    df_ord_clean = pd.DataFrame(orders_clean)

    # --- C. UTILIZATION ---
    start_m = datetime(sel_year, sel_month, 1)
    _, last_day = calendar.monthrange(sel_year, sel_month)
    end_m = datetime(sel_year, sel_month, last_day, 23, 59, 59)
    days_in_month = last_day
    
    occupancy_data = []
    
    for _, car in df_cars.iterrows():
        if not car['Active']: continue
        rented_days = 0
        revenue = 0
        if not df_ord_clean.empty:
            car_orders = df_ord_clean[df_ord_clean['Car_Code'] == car['Code']]
            for _, o in car_orders.iterrows():
                # Overlap
                latest_start = max(start_m, o['Start'])
                earliest_end = min(end_m, o['End'])
                delta = (earliest_end - latest_start).days + 1
                if delta > 0:
                    rented_days += delta
                    # Revenue pro-rated
                    total_days = (o['End'] - o['Start']).days + 1
                    daily_rate = o['Cost'] / total_days if total_days > 0 else 0
                    revenue += daily_rate * delta
        
        rented_days = min(rented_days, days_in_month)
        rate = (rented_days / days_in_month) * 100
        
        occupancy_data.append({
            'Car': car['Full_Name'], 'Type': car['Type'], 'Model': car['Model'], 
            'Rented_Days': rented_days, 'Rate': rate, 'Revenue': revenue
        })
    
    df_occupancy = pd.DataFrame(occupancy_data)
    
    # Aggregates
    total_active = len(df_cars[df_cars['Active']])
    total_rented_days = df_occupancy['Rented_Days'].sum() if not df_occupancy.empty else 0
    fleet_capacity = total_active * days_in_month
    fleet_utilization = (total_rented_days / fleet_capacity * 100) if fleet_capacity > 0 else 0

    # --- D. OWNER & FINANCIALS ---
    # Owner
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
    df_liab = pd.DataFrame(owner_liabilities)
    if not df_liab.empty:
        df_liab = df_liab.sort_values('Date')
        df_liab['Date'] = df_liab['Date'].apply(format_date)
        df_liab['Amount'] = df_liab['Amount'].apply(format_egp)

    # Monthly Financials
    def filter_df(df, year, month=None):
        if df.empty: return df
        y_col = next((c for c in df.columns if 'سنة' in c or 'Year' in c), None)
        m_col = next((c for c in df.columns if 'شهر' in c or 'Month' in c), None)
        if y_col and m_col:
            cond = df[y_col].astype(str).str.contains(str(year))
            if month: cond = cond & df[m_col].astype(str).str.contains(str(month))
            return df[cond]
        return df

    def safe_sum(df, key):
        c = next((c for c in df.columns if key in str(c)), None)
        return df[c].apply(clean_money).sum() if c else 0

    df_coll_m = filter_df(dfs['coll'], sel_year, sel_month)
    df_gen_m = filter_df(dfs['gen'], sel_year, sel_month)
    df_car_exp_m = filter_df(dfs['car_exp'], sel_year, sel_month)

    rev = safe_sum(df_coll_m, 'قيمة')
    exp_ops = safe_sum(df_gen_m, 'قيمة')
    exp_maint = safe_sum(df_car_exp_m, 'قيمة')
    total_exp = exp_ops + exp_maint + total_owner_fees
    net_profit = rev - total_exp
    margin = (net_profit / rev * 100) if rev > 0 else 0

    # --- UI ---
    st.title(f"📊 Operations Center: {sel_month} / {sel_year}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Profit", format_egp(net_profit), delta=f"{margin:.1f}% Margin")
    m2.metric("Revenue", format_egp(rev))
    m3.metric("Fleet Utilization", f"{fleet_utilization:.1f}%", delta="Occupancy")
    m4.metric("Total Expenses", format_egp(total_exp), delta_color="inverse")

    st.divider()

    tabs = st.tabs(["📅 Calendar & Util", "🚗 Fleet Analysis", "💰 Financial Trends", "⚠️ Risk Manager", "🤝 Owner Payments", "📋 Full Database", "🧠 AI Strategy"])

    # TAB 1: CALENDAR
    with tabs[0]:
        st.subheader("🗓️ Fleet Schedule (Gantt Chart)")
        if not df_ord_clean.empty:
            timeline = df_ord_clean[(df_ord_clean['Start'] <= end_m) & (df_ord_clean['End'] >= start_m)].copy()
            if not timeline.empty:
                car_map = df_cars.set_index('Code')['Full_Name'].to_dict()
                timeline['Car'] = timeline['Car_Code'].map(car_map).fillna(timeline['Car_Code'])
                fig = px.timeline(timeline, x_start="Start", x_end="End", y="Car", color="Cost", title="Bookings Timeline")
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=600, font=dict(size=14, family="Arial Black"), xaxis_range=[start_m, end_m])
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No bookings this month.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Occupancy by Type")
            if not df_occupancy.empty:
                occ_type = df_occupancy.groupby('Type')['Rate'].mean().reset_index()
                fig_occ = px.bar(occ_type, x='Type', y='Rate', color='Rate', color_continuous_scale='Greens')
                fig_occ.update_layout(height=400, font=dict(size=14))
                st.plotly_chart(fig_occ, use_container_width=True)
        with c2:
            st.subheader("Detailed Car Occupancy")
            if not df_occupancy.empty:
                disp_occ = df_occupancy[['Car', 'Rented_Days', 'Rate', 'Revenue']].sort_values('Rate', ascending=False)
                disp_occ['Rate'] = disp_occ['Rate'].apply(lambda x: f"{x:.1f}%")
                disp_occ['Revenue'] = disp_occ['Revenue'].apply(format_egp)
                st.dataframe(disp_occ, hide_index=True, use_container_width=True)

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
            st.subheader("Year Distribution")
            if not active_cars.empty:
                fig_y = px.bar(active_cars['Year'].value_counts().reset_index(), x='Year', y='count')
                st.plotly_chart(fig_y, use_container_width=True)

    # TAB 3: FINANCIALS
    with tabs[2]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Yearly Occupancy Trend")
            # Calculate full year trend
            occ_trend = []
            for m in range(1, 13):
                # Mini calc for each month
                s_m = datetime(sel_year, m, 1)
                _, ld = calendar.monthrange(sel_year, m)
                e_m = datetime(sel_year, m, ld, 23, 59, 59)
                rented = 0
                if not df_ord_clean.empty:
                    # Filter orders overlapping this month
                    ords = df_ord_clean[(df_ord_clean['Start'] <= e_m) & (df_ord_clean['End'] >= s_m)]
                    for _, o in ords.iterrows():
                        delta = (min(e_m, o['End']) - max(s_m, o['Start'])).days + 1
                        rented += max(0, delta)
                cap = total_active * ld
                rate = (rented / cap * 100) if cap > 0 else 0
                occ_trend.append({'Month': m, 'Occupancy %': rate})
            
            df_occ_trend = pd.DataFrame(occ_trend)
            fig_occ_trend = px.line(df_occ_trend, x='Month', y='Occupancy %', markers=True, title="Occupancy Rate Trend")
            fig_occ_trend.update_traces(line_color='purple', line_width=4)
            fig_occ_trend.update_layout(height=450, font=dict(size=14))
            st.plotly_chart(fig_occ_trend, use_container_width=True)

        with c2:
            st.subheader("Financial Trend")
            fin_trend = []
            for m in range(1, 13):
                r = safe_sum(filter_df(dfs['coll'], sel_year, m), 'قيمة')
                e = safe_sum(filter_df(dfs['gen'], sel_year, m), 'قيمة') + safe_sum(filter_df(dfs['car_exp'], sel_year, m), 'قيمة')
                fin_trend.append({'Month': m, 'Revenue': r, 'Expenses': e})
            df_fin_trend = pd.DataFrame(fin_trend)
            fig_fin = go.Figure()
            fig_fin.add_trace(go.Scatter(x=df_fin_trend['Month'], y=df_fin_trend['Revenue'], name='Rev', line=dict(color='green', width=4)))
            fig_fin.add_trace(go.Scatter(x=df_fin_trend['Month'], y=df_fin_trend['Expenses'], name='Exp', line=dict(color='red', width=4)))
            fig_fin.update_layout(height=450, font=dict(size=14))
            st.plotly_chart(fig_fin, use_container_width=True)

    # TAB 4: RISKS
    with tabs[3]:
        st.subheader("🚨 Risks")
        today = datetime.today()
        limits = { 'High': today+timedelta(90), 'Med': today+timedelta(180), 'Low': today+timedelta(365) }
        def get_risk(d):
            if pd.isnull(d): return None
            if today < d <= limits['High']: return "High"
            if limits['High'] < d <= limits['Med']: return "Med"
            if limits['Med'] < d <= limits['Low']: return "Low"
            return None
        risks = []
        for _, car in df_cars.iterrows():
            if not car['Active']: continue
            for k, col in [('License','License'), ('Insurance','Insurance')]:
                r = get_risk(car[col])
                if r: risks.append({'Car': car['Full_Name'], 'Type': k, 'Date': format_date(car[col]), 'Risk': r})
        if risks:
            df_r = pd.DataFrame(risks).sort_values('Date')
            c1, c2, c3 = st.columns(3)
            with c1:
                h = df_r[df_r['Risk']=='High']
                st.markdown(f"<div class='risk-high'>🔴 High ({len(h)})</div>", unsafe_allow_html=True)
                st.dataframe(h[['Car', 'Type', 'Date']], hide_index=True)
            with c2:
                m = df_r[df_r['Risk']=='Med']
                st.markdown(f"<div class='risk-med'>🟠 Medium ({len(m)})</div>", unsafe_allow_html=True)
                st.dataframe(m[['Car', 'Type', 'Date']], hide_index=True)
            with c3:
                l = df_r[df_r['Risk']=='Low']
                st.markdown(f"<div class='risk-low'>🟢 Low ({len(l)})</div>", unsafe_allow_html=True)
                st.dataframe(l[['Car', 'Type', 'Date']], hide_index=True)
        else: st.success("No risks.")

    # TAB 5: PAYMENTS
    with tabs[4]:
        st.subheader("Owner Payments")
        if not df_liab.empty: st.dataframe(df_liab, hide_index=True, use_container_width=True)
        else: st.info("No payments.")

    # TAB 6: ALL CARS
    with tabs[5]:
        st.subheader("📋 Database")
        disp = df_cars.copy()
        for c in ['License', 'Insurance', 'Contract_End', 'Pay_Start']: disp[c] = disp[c].apply(format_date)
        disp['Owner_Fee'] = disp['Owner_Fee'].apply(format_egp)
        st.dataframe(disp, hide_index=True, use_container_width=True)

    # TAB 7: AI
    with tabs[6]:
        if st.button("Generate Strategy"):
            if 'GOOGLE_API_KEY' not in st.secrets: st.error("Missing Key")
            else:
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
                prompt = f"Advisor Report: {sel_month}/{sel_year}. Rev: {rev}, Profit: {net_profit}, Util: {fleet_utilization:.1f}%. Arabic Brief."
                advisor = Agent(role='Advisor', goal='Report', backstory='Expert', llm=llm)
                task = Task(description=prompt, agent=advisor, expected_output="Summary")
                crew = Crew(agents=[advisor], tasks=[task])
                st.markdown(crew.kickoff())
