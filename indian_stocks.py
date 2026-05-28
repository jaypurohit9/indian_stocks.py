import os
import streamlit as st
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# 1. STREAMLIT UI CONFIGURATION
st.set_page_config(page_title="Dalal Street AI Analyst", page_icon="📈", layout="wide")

st.title("📈 Dalal Street AI Technical Analyst")
st.caption("A Generative AI-powered companion for Nifty 50 short-term breakout analysis.")

# Sidebar for configuration (Product focus: security and cost management)
st.sidebar.header("🔑 Authentication & Settings")
user_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Your key stays safe in your local browser session.")

# 2. DEFINE THE LIVE DATA TOOL
@tool("Fetch Indian Stock Data")
def fetch_stock_data(ticker: str) -> str:
    """Fetches the last 5 days of stock price history for a given Indian stock ticker (e.g., 'RELIANCE.NS')."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return f"No data found for {ticker}."
        data_str = hist[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()
        return f"\nRecent 5-day data for {ticker}:\n{data_str}"
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# 3. INTERACTIVE PRODUCT FEATURES
st.subheader("📋 Configure Your Analysis")
st.write("Select the Nifty 50 heavyweight stocks you want the AI agent to cross-examine:")

# User can pick which stocks to analyze (Improves User Experience/UX)
default_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TATASTEEL.NS"]
selected_stocks = st.multiselect("Target Stocks:", default_stocks, default_stocks)

# 4. TRIGGER THE AI AGENT WORKFLOW
if st.button("🚀 Run Market Analysis", type="primary"):
    if not user_api_key:
        st.error("Please enter your Gemini API Key in the sidebar to run the analysis.")
    elif len(selected_stocks) == 0:
        st.warning("Please select at least one stock to analyze.")
    else:
        with st.spinner("🔄 Brainstorming... The AI Agent is pulling live technical numbers and calculating momentum..."):
            try:
                # Configure LLM dynamically with user input key
                gemini_llm = LLM(
                    model="gemini/gemini-2.5-flash",
                    api_key=user_api_key
                )

                # Initialize Agent
                stock_expert = Agent(
                    role="NSE Technical Stock Analyst",
                    goal="Analyze live stock data to identify momentum and predict short-term breakouts.",
                    backstory="You are an expert Dalal Street day-trader. You analyze price actions, volume spikes, and recent trends on the National Stock Exchange (NSE) to spot short-term setups.",
                    verbose=True,
                    llm=gemini_llm,
                    tools=[fetch_stock_data]
                )

                # Initialize Task dynamically with selected stocks
                analysis_task = Task(
                    description=f"Use your tool to check the live 5-day data for these Indian stocks: {', '.join(selected_stocks)}. Look at the closing trends and volume changes to predict which one looks most likely to experience a short-term bullish breakout tomorrow. List your top choice and explain why based on the numbers.",
                    expected_output="A structured report highlighting the top stock for tomorrow with short-term technical reasoning.",
                    agent=stock_expert
                )

                # Assemble and Kickoff Crew
                stock_crew = Crew(
                    agents=[stock_expert],
                    tasks=[analysis_task],
                    process=Process.sequential
                )

                result = stock_crew.kickoff()
                
                # Render the Output beautifully in Markdown
                st.success("✅ Analysis Complete!")
                st.markdown("### 📊 AI Analyst Report")
                st.info(result)

            except Exception as e:
                st.error(f"An execution error occurred: {e}")