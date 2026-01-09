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

    def get_sheet_data(sheet_id, range_name, file_label):
        try:
            result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = result.get('values', [])
            if not vals: return pd.DataFrame()
            return pd.DataFrame(vals[1:], columns=vals[0])
        except Exception as e:
            st.error(f"⚠️ Error loading {file_label}: {e}")
            return pd.DataFrame()

    # ==========================================
    # 🟢 YOUR IDS (Inserted Automatically)
    # ==========================================
    ID_GEN_EXPENSES = "1hZoymf0CN1wOssc3ddQiZXxbJTdzJZBnamp_aCobl1Q"  # سجل المصروفات
    ID_CAR_EXPENSES = "1vDKKOywOEGfmLcHr4xk7KMTChHJ0_qquNopXpD81XVE"  # سجل مصروفات السيارات
    ID_CARS = "1fLr5mwDoRQ1P5g-t4uZ8mSY04xHiCSSisSWDbatx9dg"                     # السيارات
    ID_CLIENTS = "1izZeNVITKEKVCT4KUnb71uFO8pzCdpUs8t8FetAxbEg"                   # العملاء
    ID_COLLECTIONS = "1jtp-ihtAOt9NNHETZ5muiL5OA9yW3WrpBIIDAf5UAyg"     # سجل التحصيلات المالية
    ID_ORDERS = "10OK9uw2o3Yqy2akXAjmjbP_YvgGwivp14-MoeHj0cX4"                    # تسجيل الطلبات

    # Load Dataframes (Exact tab names from your files)
    df_coll = get_sheet_data(ID_COLLECTIONS, "'صفحة الإدخالات لقاعدة البيانات'!A:Z", "Collections")
    df_gen_exp = get_sheet_data(ID_GEN_EXPENSES, "'صفحة الإدخالات لقاعدة البيانات'!A:Z", "General Expenses")
    df_car_exp = get_sheet_data(ID_CAR_EXPENSES, "'صفحة الإدخالات لقاعدة البيانات'!A:Z", "Car Expenses")
    df_orders = get_sheet_data(ID_ORDERS, "'صفحة الإدخالات للإيجارات'!A:Z", "Orders")
    df_cars = get_sheet_data(ID_CARS, "'قاعدة البيانات'!A:Z", "Car Database")
    df_clients = get_sheet_data(ID_CLIENTS, "'صفحة الإدخالات لقاعدة البيانات'!A:Z", "Clients")

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
        # Combining Model and Color
        df_cars['Full_Name'] = df_cars['الطراز'] + " - " + df_cars['اللون']
        df_cars['No.'] = df_cars['No.'].astype(str)
        car_map = df_cars.set_index('No.')['Full_Name'].to_dict()

    # --- CALCULATIONS ---
    total_rev = df_coll['قيمة التحصيل'].sum() if not df_coll.empty else 0
    total_gen_ops = df_gen_exp['قيمة المصروف'].sum() if not df_gen_exp.empty else 0
    total_car_maint = df_car_exp['قيمة المصروف'].sum() if not df_car_exp.empty else 0
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
                df_orders['Total_Cost'] = pd.to_numeric(df_orders['التكلفة الإجمالية'], errors='coerce').fillna(0)
                df_orders['Car_ID_Clean'] = df_orders['كود السيارة'].astype(str).str.strip()
                car_rev = df_orders.groupby('Car_ID_Clean')['Total_Cost'].sum().reset_index(name='Revenue')
            else:
                car_rev = pd.DataFrame(columns=['Car_ID_Clean', 'Revenue'])

            if not df_car_exp.empty:
                car_id_col = [c for c in df_car_exp.columns if 'كود السيارة' in c][0]
                df_car_exp['Car_ID_Clean'] = df_car_exp[car_id_col].astype(str).str.strip()
                car_cost = df_car_exp.groupby('Car_ID_Clean')['قيمة المصروف'].sum().reset_index(name='Expense')
            else:
                car_cost = pd.DataFrame(columns=['Car_ID_Clean', 'Expense'])

            fleet = pd.merge(car_rev, car_cost, on='Car_ID_Clean', how='outer').fillna(0)
            fleet['Net_Profit'] = fleet['Revenue'] - fleet['Expense']
            fleet['Car_Name'] = fleet['Car_ID_Clean'].map(car_map).fillna(fleet['Car_ID_Clean'])
            fleet = fleet.sort_values('Net_Profit', ascending=False)
            
            st.dataframe(
                fleet[['Car_Name', 'Revenue', 'Expense', 'Net_Profit']],
                use_container_width=True,
                column_config={"Net_Profit": st.column_config.ProgressColumn("صافي الربح", format="%d EGP", min_value=0, max_value=int(fleet['Net_Profit'].max()))}
            )

        with c2:
            st.subheader("أهم العملاء (Top Clients)")
            if not df_orders.empty and not df_clients.empty and 'كود العميل' in df_orders.columns:
                # Merge Orders with Client Names
                top_clients = df_orders['كود العميل'].value_counts().head(5).reset_index()
                top_clients.columns = ['Client_ID', 'Rentals']
                
                # Try to map names if possible
                df_clients['No.'] = df_clients['No.'].astype(str)
                client_map = df_clients.set_index('No.')['الاسم الأول'].to_dict()
                top_clients['Name'] = top_clients['Client_ID'].astype(str).map(client_map)
                
                st.table(top_clients[['Name', 'Rentals']])
            else:
                st.info("لا توجد بيانات كافية للعملاء")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("الإيرادات vs المصروفات")
            fin_df = pd.DataFrame({'Type':['Revenue', 'Total Expense'], 'Value':[total_rev, total_expenses]})
            st.plotly_chart(px.pie(fin_df, names='Type', values='Value', hole=0.4), use_container_width=True)
            
        with c2:
            st.subheader("توزيع المصروفات")
            all_exp = pd.concat([
                df_gen_exp[['بيان المصروف', 'قيمة المصروف']].rename(columns={'بيان المصروف':'Item', 'قيمة المصروف':'Value'}),
                df_car_exp[['نوع المصروف', 'قيمة المصروف']].rename(columns={'نوع المصروف':'Item', 'قيمة المصروف':'Value'})
            ])
            if not all_exp.empty:
                st.plotly_chart(px.treemap(all_exp, path=['Item'], values='Value'), use_container_width=True)

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
