import os
import requests
import streamlit as st

st.title("My Personal AI Assistant")
st.write("這個網頁會把你的問題傳送到我自己訓練的 AI API。")

# 之後部署 API 後，把網址填在這裡
API_URL = os.getenv("API_URL", "https://請改成你的API網址/ask")

question = st.text_input("請輸入你的問題：")

if st.button("Ask"):
    if not question.strip():
        st.warning("請先輸入問題。")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=10,
            )
            response.raise_for_status()
            st.success(response.json()["answer"])
        except requests.RequestException:
            st.error("無法連線到 AI API。請確認 API 是否已經部署。")
