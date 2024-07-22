import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Stocks List",
    page_icon="📑",
)

st.title('Stocks List')

st.sidebar.header('Stocks List')

data = pd.read_csv('Daftar Saham.csv')

st.subheader('List of Indonesia Stocks')
st.dataframe(data, width=800)
