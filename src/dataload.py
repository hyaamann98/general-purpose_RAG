from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path

class DataLoad():

    def __init__(self, source: str):
        """
        source : local / s3
        """
        self.source = source

    def load(self, path = None) -> list[Document]:
        if path is None:
            raise ValueError("Please enter input Document path !!!")

        if self.source == "local":
            text_list = self._load_local(path)
        elif self.source == "s3":
            self._load_s3(path)
        
        return text_list
        
    def _load_local(self, path) -> list[Document]:
        # documents
        file_path_list = self._list_files_local(path)        
        text_list = []
        # flie loop
        for file_path in file_path_list:
            # .pdf    
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            # .txt
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue

            text = loader.load()
            text_list.extend(text)
        return text_list

    def _list_files_local(self, path: str) -> list[str]:
        file_list = [str(p) for p in Path(path).rglob("*") if p.is_file()]
        return file_list

    def _load_s3(self, path):
        # TODO: S3内のファイルを繰り返し呼び出す処理を入れる
        # TODO: テキストロード
        pass

if __name__ == '__main__':
    source = "local"
    path = "./../sample_dat"

    dataload = DataLoad(source)
    text_list = dataload.load(path)
    print(text_list)
    pass
