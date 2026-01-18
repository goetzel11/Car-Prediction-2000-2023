import streamlit as st
import eda1, PREDICT

with st.sidebar:
    st.title('Page Navigation')
    page = st.selectbox('Pilih Halaman',
                        ['eda1', 'PREDICT'])
    
if page == 'eda1':
    eda1.run()
else:
    PREDICT.run()