import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="社内文書 RAG チャット", page_icon="📄")
st.title("📄 社内文書 RAG チャット")

# ---- サイドバー：文書アップロード ----
with st.sidebar:
    st.header("文書の登録")
    uploaded_file = st.file_uploader(
        "ファイルを選択（.txt / .pdf）",
        type=["txt", "pdf"],
    )
    if st.button("アップロード", disabled=uploaded_file is None):
        with st.spinner("処理中..."):
            res = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
            )
        if res.ok:
            st.success(res.json().get("message", "完了しました。"))
        else:
            st.error(f"エラー: {res.text}")

    st.divider()
    st.header("登録済みファイル一覧")
    if st.button("一覧を更新"):
        st.session_state.pop("documents", None)
    if "documents" not in st.session_state:
        try:
            res = requests.get(f"{API_URL}/documents", timeout=10)
            if res.ok:
                st.session_state.documents = res.json()
            else:
                st.session_state.documents = None
        except requests.exceptions.ConnectionError:
            st.session_state.documents = None
    doc_data = st.session_state.get("documents")
    if doc_data is None:
        st.warning("APIサーバーに接続できません。")
    elif doc_data["count"] == 0:
        st.info("登録されたファイルはありません。")
    else:
        st.caption(f"{doc_data['count']} 件")
        for path in doc_data["documents"]:
            st.write(f"- {os.path.basename(path)}")

    st.divider()
    k = st.slider("参照するチャンク数（k）", min_value=1, max_value=10, value=5)

# ---- チャット履歴の初期化 ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- 過去のチャットを表示 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("参照元ファイル"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

# ---- 入力欄 ----
if question := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            try:
                res = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "k": k},
                    timeout=300,
                )
                if res.ok:
                    data = res.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                else:
                    answer = f"エラーが発生しました: {res.text}"
                    sources = []
            except requests.exceptions.ConnectionError:
                answer = "APIサーバーに接続できません。サーバーが起動しているか確認してください。"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("参照元ファイル"):
                for s in sources:
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
