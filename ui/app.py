import streamlit as st
import style

st.set_page_config(page_title="RAG Lab", page_icon=":material/hub:", layout="wide")
style.inject()

pages = [
    st.Page("experiments.py", title="Experiments", icon=":material/folder_managed:"),
    st.Page("chat.py", title="Chatbot", icon=":material/chat:", default=True),
]
st.navigation(pages).run()
