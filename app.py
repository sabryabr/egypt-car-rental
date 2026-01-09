import streamlit as st
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from googleapiclient.discovery import build
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Egypt Rental CEO Dashboard", layout="wide", page_icon="🚗")

st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    .stMetric { background-color: #f8f9fa; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONNECT TO GOOGLE SHEETS ---
@st.cache_data(ttl=600)
def load_data():
    # Load secrets
    if "gcp_service_account" not in st.secrets:
        st.error("Missing Google Cloud Secrets!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    def get_sheet_data(sheet_id, range_name):
        try:
            result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
            vals = result.get('values', [])
            if not vals: return pd.DataFrame()
            return pd.DataFrame(vals[1:], columns=vals[0])
        except Exception as e:
            st.error(f"Error reading sheet: {e}")
            return pd.DataFrame()

    # REPLACE THESE WITH YOUR ACTUAL IDS
    ID_FINANCE = "YOUR_COLLECTIONS_AND_EXPENSES_SHEET_ID" 
    ID_ORDERS = "YOUR_ORDERS_SHEET_ID"
    
    # Adjust tab names (e.g. 'Sheet1') to match your actual Arabic tab names exactly
    df_coll = get_sheet_data(ID_FINANCE, "'سجل التحصيلات المالية'!A:Z") 
    df_exp = get_sheet_data(ID_FINANCE, "'سجل المصروفات'!A:Z")
    df_orders = get_sheet_data(ID_ORDERS, "'صفحة الإدخالات للإيجارات'!A:Z")

    return df_coll, df_exp, df_orders

# --- 3. MAIN APP ---
try:
    df_coll, df_exp, df_orders = load_data()

    # --- DATA CLEANING ---
    for df in [df_coll, df_exp]:
        cols = [c for c in df.columns if 'قيمة' in c]
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # --- DASHBOARD ---
    st.title("🇪🇬 لوحة القيادة (Powered by Gemini)")
    
    total_rev = df_coll['قيمة التحصيل'].sum() if 'قيمة التحصيل' in df_coll.columns else 0
    total_exp = df_exp['قيمة المصروف'].sum() if 'قيمة المصروف' in df_exp.columns else 0
    net_profit = total_rev - total_exp

    k1, k2, k3 = st.columns(3)
    k1.metric("صافي الربح", f"{net_profit:,.0f} EGP", delta="Live")
    k2.metric("الإيرادات", f"{total_rev:,.0f} EGP")
    k3.metric("المصروفات", f"{total_exp:,.0f} EGP", delta_color="inverse")

    st.divider()

    # --- 4. GEMINI AI INTELLIGENCE ---
    st.subheader("🧠 التحليل الذكي (Gemini AI)")
    
    if st.button("Run Daily Analysis (مجاني)"):
        if 'GOOGLE_API_KEY' not in st.secrets:
            st.error("Please add GOOGLE_API_KEY to Streamlit Secrets.")
        else:
            with st.spinner("Gemini is analyzing your sheets..."):
                # 1. Setup Gemini
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                
                # We use the free 'gemini-1.5-flash' model which is fast and free
                gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.3,
                    google_api_key=st.secrets["GOOGLE_API_KEY"]
                )

                # 2. Summarize Data
                fin_summary = f"Revenue: {total_rev}, Expenses: {total_exp}, Net: {net_profit}"
                
                # 3. Create Agent with Gemini
                cfo = Agent(
                    role='CFO', 
                    goal='Analyze financial health', 
                    backstory='Strict accountant', 
                    llm=gemini_llm,  # <--- WE TELL IT TO USE GEMINI HERE
                    verbose=True,
                    allow_delegation=False
                )
                
                # 4. Create Task
                task = Task(
                    description=f"Analyze this data: {fin_summary}. Give 3 bullet points in Arabic about the financial status.",
                    agent=cfo,
                    expected_output="Bullet points in Arabic"
                )
                
                # 5. Run
                crew = Crew(agents=[cfo], tasks=[task])
                result = crew.kickoff()
                
                st.success("تم التحليل:")
                st.markdown(result)

    # --- VISUALS ---
    st.subheader("📊 الرسوم البيانية")
    tab1, tab2 = st.tabs(["الربحية", "الأسطول"])
    with tab1:
        chart_data = pd.DataFrame({'Type':['Revenue','Expense'], 'Value':[total_rev, total_exp]})
        st.plotly_chart(px.bar(chart_data, x='Type', y='Value', color='Type'), use_container_width=True)

except Exception as e:
    st.warning("Please check your Google Sheet connections.")
    st.error(f"Error: {e}")
