from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class MakeChunk():

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,                    # チャンクの最大サイズ
            chunk_overlap=chunk_overlap,              # 重複させる文字数
            separators=[
                "\n\n",  # 段落区切り（最優先）
                "\n",    # 改行
                "。",    # 日本語の句点
                "!",    # 感嘆符
                "？",    # 疑問符
                " ",     # 空白
                "",      # 文字単位（最後の手段）
            ],
            length_function=len,                      # 長さの測定方法
        )

    def text_split(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        return chunks
    
if __name__ == "__main__":
    from langchain_core.documents import Document

    make_chunk = MakeChunk()

    documents = [Document(page_content="私は、日本人の男です。兵庫県川西市出身で、大阪府大阪市に住んでいます。職業はエンジニアです。")]
    chunks = make_chunk.text_split(documents)
    print(type(chunks))
    # print(chunks)

    # 分割結果の確認
    # import streamlit as st
    # st.write(f"📊 分割統計:")
    # st.write(f"- 元文書: {len(documents):,}文字")
    # st.write(f"- チャンク数: {len(chunks)}")
    # st.write(f"- 平均チャンクサイズ: {sum(len(c) for c in chunks) // len(chunks)}文字")
