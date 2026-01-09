import streamlit as st
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from googleapiclient.discovery import build
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Egypt Rental Command Center", layout="wide", page_icon="🚘")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    .stMetric { background-color: #f8f9fa; border-radius: 10px; padding: 10px; border: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONNECT TO GOOGLE SHEETS ---
@st.cache_data(ttl=600)
def load_data():
    if "gcp_service_account" not in st.secrets:
        st.error("Missing Secrets. Please add your Google Cloud JSON in Streamlit settings.")
        return None

    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    # --- SMART LOADER FUNCTION ---
    def get_sheet_data(sheet_id, range_name, file_label):
        try:
            result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = result.get('values', [])
            if not vals: return pd.DataFrame()

            # --- SMART HEADER DETECTION ---
            # We look for the row that contains specific keywords to identify it as the Header Row
            # This skips Title rows automatically
            header_index = 0
            header_found = False
            
            # Keywords likely to be in headers
            keywords = ['No.', 'كود', 'الاسم', 'Name', 'Code', 'المحافظة', 'Date', 'Type']
            
            for i, row in enumerate(vals[:10]): # Scan first 10 rows
                # Check if any keyword matches any cell in this row
                row_str = [str(c).strip() for c in row]
                if any(k in row_str for k in keywords):
                    header_index = i
                    header_found = True
                    break
            
            if not header_found:
                # Fallback: Use row 0 if no keywords found, but assume it might be data if it's messy
                header_index = 0

            # Separate Headers and Data
            headers = vals[header_index]
            data = vals[header_index+1:]

            # --- FIX COLUMN MISMATCH ---
            # Ensure headers list has unique names (pandas hates duplicates)
            unique_headers = []
            seen = set()
            for h in headers:
                h_str = str(h).strip()
                if h_str in seen or h_str == "":
                    h_str = f"{h_str}_{len(seen)}" if h_str else f"Col_{len(seen)}"
                seen.add(h_str)
                unique_headers.append(h_str)
            
            # Ensure all data rows have same length as headers (Pad with None)
            max_cols = len(unique_headers)
            clean_data = []
            for row in data:
                # Truncate if too long, Pad if too short
                if len(row) > max_cols:
                    clean_data.append(row[:max_cols])
                else:
                    clean_data.append(row + [None] * (max_cols - len(row)))

            df = pd.DataFrame(clean_data, columns=unique_headers)
            return df

        except Exception as e:
            st.error(f"⚠️ Error loading {file_label}: {e}")
            return pd.DataFrame()

    # ==========================================
    # 🟢 YOUR IDS (Please verify these match your latest converted files)
    # ==========================================
    ID_GEN_EXPENSES = "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q"
    ID_CAR_EXPENSES = "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE"
    ID_CARS = "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg"
    ID_CLIENTS = "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg"
    ID_COLLECTIONS = "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg"
    ID_ORDERS = "16mLWxdxpV6DDaGfeLf-t1XDx25H4rVEbtx_hE88nF7A"

    # Load Dataframes
    df_coll = get_sheet_data(ID_COLLECTIONS, "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", "Collections")
    df_gen_exp = get_sheet_data(ID_GEN_EXPENSES, "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", "General Expenses")
    df_car_exp = get_sheet_data(ID_CAR_EXPENSES, "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", "Car Expenses")
    df_orders = get_sheet_data(ID_ORDERS, "'صفحة الإدخالات للإيجارات'!A:ZZ", "Orders")
    df_cars = get_sheet_data(ID_CARS, "'قاعدة البيانات'!A:ZZ", "Car Database")
    df_clients = get_sheet_data(ID_CLIENTS, "'صفحة الإدخالات لقاعدة البيانات'!A:ZZ", "Clients")

    return df_coll, df_gen_exp, df_car_exp, df_orders, df_cars, df_clients

# --- 3. MAIN APPLICATION ---
data = load_data()

if data:
    df_coll, df_gen_exp, df_car_exp, df_orders, df_cars, df_clients = data

    # --- DATA CLEANING & PREP ---
    # 1. Convert Money Columns to Numbers
    for df in [df_coll, df_gen_exp, df_car_exp]:
        if not df.empty:
            cols = [c for c in df.columns if 'قيمة' in c]
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 2. Map Car IDs to Real Names
    car_map = {}
    if not df_cars.empty:
        # Search for columns that look like Model/Color regardless of exact name
        model_col = next((c for c in df_cars.columns if 'طراز' in c or 'Model' in c), None)
        color_col = next((c for c in df_cars.columns if 'لون' in c or 'Color' in c), None)
        id_col = next((c for c in df_cars.columns if 'No' in c or 'كود' in c), None)

        if model_col and id_col:
            df_cars['Full_Name'] = df_cars[model_col].astype(str)
            if color_col:
                df_cars['Full_Name'] += " - " + df_cars[color_col].astype(str)
            
            df_cars[id_col] = df_cars[id_col].astype(str).str.strip()
            car_map = df_cars.set_index(id_col)['Full_Name'].to_dict()

    # --- CALCULATIONS ---
    total_rev = df_coll['قيمة التحصيل'].sum() if not df_coll.empty and 'قيمة التحصيل' in df_coll.columns else 0
    total_gen_ops = df_gen_exp['قيمة المصروف'].sum() if not df_gen_exp.empty and 'قيمة المصروف' in df_gen_exp.columns else 0
    total_car_maint = df_car_exp['قيمة المصروف'].sum() if not df_car_exp.empty and 'قيمة المصروف' in df_car_exp.columns else 0
    total_expenses = total_gen_ops + total_car_maint
    net_profit = total_rev - total_expenses
    
    # --- DASHBOARD HEADER ---
    st.title("📟 مركز القيادة: تأجير السيارات (Full Operations)")
    
    # --- KPIS ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 صافي الربح (Net Profit)", f"{net_profit:,.0f} EGP", delta="Live")
    k2.metric("📥 الإيرادات (Revenue)", f"{total_rev:,.0f} EGP")
    k3.metric("🛠 مصروفات السيارات", f"{total_car_maint:,.0f} EGP", delta_color="inverse")
    k4.metric("🏢 مصروفات إدارية", f"{total_gen_ops:,.0f} EGP", delta_color="inverse")

    st.divider()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🚗 أداء الأسطول (Fleet)", "📉 التحليل المالي", "🧠 المدير الذكي (AI)"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("تحليل ربحية السيارات (Profitability per Car)")
            
            if not df_orders.empty:
                # Find columns safely
                cost_col = next((c for c in df_orders.columns if 'إجمال' in c or 'Total' in c), None)
                car_code_col = next((c for c in df_orders.columns if 'كود السيارة' in c or 'Car' in c), None)
                
                if cost_col and car_code_col:
                    df_orders['Total_Cost_Num'] = pd.to_numeric(df_orders[cost_col], errors='coerce').fillna(0)
                    df_orders['Car_ID_Clean'] = df_orders[car_code_col].astype(str).str.strip()
                    car_rev = df_orders.groupby('Car_ID_Clean')['Total_Cost_Num'].sum().reset_index(name='Revenue')
                else:
                    st.warning(f"Could not find 'Total Cost' or 'Car Code' columns in Orders. Found: {list(df_orders.columns)}")
                    car_rev = pd.DataFrame(columns=['Car_ID_Clean', 'Revenue'])
            else:
                car_rev = pd.DataFrame(columns=['Car_ID_Clean', 'Revenue'])

            if not df_car_exp.empty:
                car_id_col = next((c for c in df_car_exp.columns if 'كود السيارة' in c), None)
                val_col = next((c for c in df_car_exp.columns if 'قيمة' in c), None)

                if car_id_col and val_col:
                    df_car_exp['Car_ID_Clean'] = df_car_exp[car_id_col].astype(str).str.strip()
                    df_car_exp['Val_Num'] = pd.to_numeric(df_car_exp[val_col], errors='coerce').fillna(0)
                    car_cost = df_car_exp.groupby('Car_ID_Clean')['Val_Num'].sum().reset_index(name='Expense')
                else:
                    car_cost = pd.DataFrame(columns=['Car_ID_Clean', 'Expense'])
            else:
                car_cost = pd.DataFrame(columns=['Car_ID_Clean', 'Expense'])

            fleet = pd.merge(car_rev, car_cost, on='Car_ID_Clean', how='outer').fillna(0)
            fleet['Net_Profit'] = fleet['Revenue'] - fleet['Expense']
            fleet['Car_Name'] = fleet['Car_ID_Clean'].map(car_map).fillna(fleet['Car_ID_Clean'])
            fleet = fleet.sort_values('Net_Profit', ascending=False)
            
            st.dataframe(
                fleet[['Car_Name', 'Revenue', 'Expense', 'Net_Profit']],
                use_container_width=True,
                column_config={"Net_Profit": st.column_config.ProgressColumn("صافي الربح", format="%d EGP", min_value=0, max_value=int(fleet['Net_Profit'].max()) if not fleet.empty else 0)}
            )

        with c2:
            st.subheader("أهم العملاء (Top Clients)")
            client_col = next((c for c in df_orders.columns if 'كود العميل' in c or 'Client' in c), None)
            
            if not df_orders.empty and client_col:
                top_clients = df_orders[client_col].value_counts().head(5).reset_index()
                top_clients.columns = ['Client_ID', 'Rentals']
                
                # Check client mapping
                c_id_col = next((c for c in df_clients.columns if 'No' in c), None)
                c_name_col = next((c for c in df_clients.columns if 'اسم' in c or 'Name' in c), None)

                if not df_clients.empty and c_id_col and c_name_col:
                    df_clients[c_id_col] = df_clients[c_id_col].astype(str).str.strip()
                    client_map = df_clients.set_index(c_id_col)[c_name_col].to_dict()
                    top_clients['Name'] = top_clients['Client_ID'].astype(str).map(client_map).fillna(top_clients['Client_ID'])
                else:
                    top_clients['Name'] = top_clients['Client_ID']
                
                st.table(top_clients[['Name', 'Rentals']])
            else:
                st.info("No sufficient client data.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("الإيرادات vs المصروفات")
            fin_df = pd.DataFrame({'Type':['Revenue', 'Total Expense'], 'Value':[total_rev, total_expenses]})
            st.plotly_chart(px.pie(fin_df, names='Type', values='Value', hole=0.4), use_container_width=True)
            
        with c2:
            st.subheader("توزيع المصروفات")
            # Safe rename
            g_item = next((c for c in df_gen_exp.columns if 'بيان' in c), 'Item')
            g_val = next((c for c in df_gen_exp.columns if 'قيمة' in c), 'Value')
            c_item = next((c for c in df_car_exp.columns if 'نوع' in c), 'Item')
            c_val = next((c for c in df_car_exp.columns if 'قيمة' in c), 'Value')

            all_exp = pd.concat([
                df_gen_exp[[g_item, g_val]].rename(columns={g_item:'Item', g_val:'Value'}) if not df_gen_exp.empty else pd.DataFrame(),
                df_car_exp[[c_item, c_val]].rename(columns={c_item:'Item', c_val:'Value'}) if not df_car_exp.empty else pd.DataFrame()
            ])
            # Clean values
            all_exp['Value'] = pd.to_numeric(all_exp['Value'], errors='coerce').fillna(0)
            
            if not all_exp.empty and all_exp['Value'].sum() > 0:
                st.plotly_chart(px.treemap(all_exp, path=['Item'], values='Value'), use_container_width=True)
            else:
                st.info("No expenses recorded yet.")

    with tab3:
        st.subheader("تقرير المدير العام (Gemini AI)")
        if st.button("اطلب تحليل الذكاء الاصطناعي"):
            if 'GOOGLE_API_KEY' not in st.secrets:
                st.error("Please add GOOGLE_API_KEY to secrets.")
            else:
                with st.spinner("Analyzing full fleet data..."):
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

                    top_car = fleet.iloc[0]['Car_Name'] if not fleet.empty else "N/A"
                    worst_car = fleet.iloc[-1]['Car_Name'] if not fleet.empty else "N/A"
                    
                    summary = f"""
                    Financials: Net Profit {net_profit}, Revenue {total_rev}, Expenses {total_expenses}.
                    Fleet: Best Car is {top_car}, Worst performing car is {worst_car}.
                    """
                    
                    advisor = Agent(role='Strategic Advisor', goal='Optimize Business', backstory='Expert Consultant', llm=llm)
                    task = Task(description=f"Analyze this: {summary}. Write a professional daily briefing in Arabic for the owner. Focus on profit and car performance.", agent=advisor, expected_output="Arabic Briefing")
                    
                    crew = Crew(agents=[advisor], tasks=[task])
                    res = crew.kickoff()
                    st.markdown(res)
