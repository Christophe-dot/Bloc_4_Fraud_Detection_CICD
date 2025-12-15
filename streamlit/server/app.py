from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import os
from urllib.error import URLError

@st.cache_data
def get_data() -> pd.DataFrame:
    conn = st.connection(
        "postgresql",
        type="sql",
        url=os.environ['BACKEND_STORE_URI']
    )
    # Today
    #df = conn.query("SELECT * FROM transactions WHERE trans_date_trans_time > TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD') || ' 00:00:00'")

    # Yesterday
    #df = conn.query("SELECT * FROM transactions WHERE trans_date_trans_time >= TO_CHAR(CURRENT_DATE - INTERVAL '1 day', 'YYYY-MM-DD') || ' 00:00:00' AND trans_date_trans_time < TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD') || ' 00:00:00'")

    # ALL
    df = conn.query("SELECT * FROM transactions")
    return df

# TODAY= pd.Timestamp.now().strftime('%Y-%m-%d')
# st.title(f"TRANSACTIONS REPORT\n Today is: {TODAY}")

# YESTERDAY = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
# st.title(f"TRANSACTIONS REPORT\n Yesterday was: {YESTERDAY}")

# ALL
st.title(f"TRANSACTIONS REPORT")
 
try:
    df = get_data()
    st.sidebar.header("Date Selection")

    df['date'] = pd.to_datetime(df['trans_date_trans_time']).dt.date
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    mask = (df['date'] > min_date) & (df['date'] <= max_date)
    df_filtered = df.loc[mask]

    st.dataframe(data = df_filtered, width="content", hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total transactions: ", value=len(df))
        st.metric(label="Total amount of transactions: ", value=f"${df['amt'].sum():,.2f}")
    with col2:
        st.metric(label="Number of fraud cases detected: ", value=len(df[df['classification'] == 1]))
        st.metric(label="Total amount of fraud cases: ", value=f"${df[df['classification'] == 1]['amt'].sum():,.2f}")

except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")

