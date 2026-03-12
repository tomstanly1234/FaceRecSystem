# src/streamlit_app.py
import streamlit as st
import pandas as pd
import sqlite3
import requests
from datetime import datetime, timedelta
#import plotly.express as px

DB_PATH = "../db/attendance.db"

st.set_page_config(page_title="Smart Attendance Dashboard", layout="wide")
st.title("🎓 Smart Attendance Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")
refresh_btn = st.sidebar.button("🔄 Refresh Data")


# Database connection function
def get_attendance_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


# Main dashboard
df = get_attendance_data()

if df.empty:
    st.warning("No attendance data yet. Run the real-time recognition script.")
else:
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Unique Persons", df['name'].nunique())

    with col3:
        today_count = len(df[df['timestamp'].dt.date == datetime.today().date()])
        st.metric("Today's Attendance", today_count)

    with col4:
        recent_count = len(df[df['timestamp'] > datetime.now() - timedelta(hours=24)])
        st.metric("Last 24h", recent_count)

    # Data table
    st.subheader("📊 Attendance Records")
    st.dataframe(df, use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attendance by Person")
        attendance_counts = df['name'].value_counts()
        st.bar_chart(attendance_counts)

    with col2:
        st.subheader("Recent Activity")
        # Daily attendance trend
        daily_counts = df.set_index('timestamp').resample('D').size()
        st.line_chart(daily_counts)

    # Export option
    st.subheader("📤 Export Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
**How to use:**
1. Run `collect_dataset.py` to collect face data
2. Run `generate_embeddings.py` to create embeddings  
3. Run `train_classifier.py` to train the model
4. Run `realtime_recognition.py` for attendance tracking
5. View results here!
""")

st.info("💡 Press 'r' in webcam window to manually record attendance, or 'q' to quit.")