from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class Embedding:
    """
    This class is for Documents/Query Embedding.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[Document]):
        texts = [doc.page_content for doc in documents]
        embedded_list = self.model.encode(texts)
        return embedded_list

    def embed_query(self, text: str) -> list[float]:
        embedded = self.model.encode(text)
        return embedded

if __name__ == '__main__':

    print("embedding.pyのmain実行します。")

    sentences = [
        Document(page_content="私は自然言語処理が好きです。"),
        Document(page_content="機械学習の応用範囲は広がっている。"),
    ]

    embedding = Embedding(model_name="intfloat/multilingual-e5-base")
    print("> embed_documentsメソッドの実行")
    embed_docs = embedding.embed_documents(sentences)
    
    print("> embed_queryメソッドの実行")
    # print(embedding.embed_query(sentences[0]))
    embed_query = embedding.embed_query(sentences[0])
    
    print("> 類似度の計算")
    similarities = embedding.model.similarity(embed_docs[1], embed_query)
    print(similarities)
