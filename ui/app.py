import json
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ドキュメント検索チャット", page_icon="📄")
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
st.title("📄 ドキュメント検索チャット")

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
    k = st.slider("参照するチャンク数（k）", min_value=1, max_value=10, value=3)

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
                    st.write(f"- {os.path.basename(s)}")

# ---- 入力欄 ----
if question := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources_holder = []

        def token_stream():
            try:
                with requests.post(
                    f"{API_URL}/query/stream",
                    json={"question": question, "k": k},
                    stream=True,
                    timeout=300,
                ) as r:
                    if not r.ok:
                        yield f"エラーが発生しました: {r.text}"
                        return
                    buffer = ""
                    marker = "__SOURCES__"
                    for raw in r.iter_content(chunk_size=None, decode_unicode=True):
                        buffer += raw
                        if marker in buffer:
                            idx = buffer.index(marker)
                            text_before = buffer[:idx]
                            if text_before:
                                yield text_before
                            try:
                                sources_holder.extend(
                                    json.loads(buffer[idx + len(marker):])
                                )
                            except Exception:
                                pass
                            return
                        # マーカーが途中で分断される可能性を考慮して末尾を保持
                        safe_end = max(0, len(buffer) - len(marker))
                        if safe_end > 0:
                            yield buffer[:safe_end]
                            buffer = buffer[safe_end:]
                    if buffer:
                        yield buffer
            except requests.exceptions.ConnectionError:
                yield "APIサーバーに接続できません。サーバーが起動しているか確認してください。"

        thinking = st.empty()
        thinking.markdown("検索中...")

        def stream_with_indicator():
            first = True
            for token in token_stream():
                if first:
                    thinking.empty()
                    first = False
                yield token

        answer = st.write_stream(stream_with_indicator())
        sources = sources_holder

        if sources:
            with st.expander("参照元ファイル"):
                for s in sources:
                    st.write(f"- {os.path.basename(s)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
