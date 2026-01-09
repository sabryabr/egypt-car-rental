import streamlit as st
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from googleapiclient.discovery import build
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Egypt Rental CEO Dashboard", layout="wide", page_icon="🚗")

# Custom CSS for Arabic
st.markdown("""
<style>
    .main { direction: rtl; text-align: right; }
    h1, h2, h3, p, div { font-family: 'Cairo', sans-serif; }
    .stMetric { background-color: #f8f9fa; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. CONNECT TO GOOGLE SHEETS ---
@st.cache_data(ttl=600) # Auto-refresh every 10 mins
def load_data():
    # Load secrets from Streamlit Cloud
    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    def get_sheet_data(sheet_id, range_name):
        result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
        vals = result.get('values', [])
        if not vals: return pd.DataFrame()
        return pd.DataFrame(vals[1:], columns=vals[0])

    # --- INPUT YOUR SHEET IDs HERE ---
    # Example ID: 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms (It is the long code in your URL)
    ID_FINANCE = "YOUR_COLLECTIONS_AND_EXPENSES_SHEET_ID" 
    ID_ORDERS = "YOUR_ORDERS_SHEET_ID"
    
    # Fetch Data (Adjust 'Sheet1' to your actual tab names like 'قاعدة البيانات')
    # Note: I am assuming you might have combined files or separate ones. 
    # Use the ID of the specific file for each dataframe.
    df_coll = get_sheet_data(ID_FINANCE, "'سجل التحصيلات المالية'!A:Z") 
    df_exp = get_sheet_data(ID_FINANCE, "'سجل المصروفات'!A:Z")
    df_orders = get_sheet_data(ID_ORDERS, "'صفحة الإدخالات للإيجارات'!A:Z")

    return df_coll, df_exp, df_orders

# --- 3. MAIN APP ---
try:
    if 'OPENAI_API_KEY' in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    df_coll, df_exp, df_orders = load_data()

    # --- DATA CLEANING (Automatic) ---
    # Convert numbers from text to float
    for df in [df_coll, df_exp]:
        # Look for columns like "قيمة" or "Value"
        cols = [c for c in df.columns if 'قيمة' in c]
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # --- DASHBOARD ---
    st.title("🇪🇬 لوحة القيادة الآلية (Real-Time)")
    
    # Calculate KPIs
    total_rev = df_coll['قيمة التحصيل'].sum() if 'قيمة التحصيل' in df_coll.columns else 0
    total_exp = df_exp['قيمة المصروف'].sum() if 'قيمة المصروف' in df_exp.columns else 0
    net_profit = total_rev - total_exp

    # Top Metrics
    k1, k2, k3 = st.columns(3)
    k1.metric("صافي الربح (Net Profit)", f"{net_profit:,.0f} EGP", delta="Live")
    k2.metric("الإيرادات (Revenue)", f"{total_rev:,.0f} EGP")
    k3.metric("المصروفات (Expenses)", f"{total_exp:,.0f} EGP", delta_color="inverse")

    st.divider()

    # --- 4. CREW AI INTELLIGENCE ---
    st.subheader("🧠 التحليل الذكي (AI Analysis)")
    
    if st.button("Run Daily Analysis"):
        if 'OPENAI_API_KEY' not in st.secrets:
            st.error("Please add OPENAI_API_KEY to Streamlit Secrets.")
        else:
            with st.spinner("The AI Crew is reading your live sheets..."):
                # Summarize data for the AI (to save tokens)
                fin_summary = f"Revenue: {total_rev}, Expenses: {total_exp}, Net: {net_profit}"
                
                # Agents
                cfo = Agent(role='CFO', goal='Analyze financial health', backstory='Strict accountant', verbose=True)
                
                # Task
                task = Task(
                    description=f"Analyze this car rental data: {fin_summary}. Give 3 bullet points on financial health and 1 warning.",
                    agent=cfo,
                    expected_output="A list of bullet points."
                )
                
                crew = Crew(agents=[cfo], tasks=[task])
                result = crew.kickoff()
                
                st.success("Analysis Ready:")
                st.info(result)

    # --- VISUALS ---
    st.subheader("📊 الرسوم البيانية")
    tab1, tab2 = st.tabs(["Profitability", "Fleet"])
    
    with tab1:
        # Simple Bar Chart
        chart_data = pd.DataFrame({'Type':['Revenue','Expense'], 'Value':[total_rev, total_exp]})
        st.plotly_chart(px.bar(chart_data, x='Type', y='Value', color='Type'), use_container_width=True)

except Exception as e:
    st.warning("Please connect your Google Sheets in the code!")
    st.error(f"Error details: {e}")
