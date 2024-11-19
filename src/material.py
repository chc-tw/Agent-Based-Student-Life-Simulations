from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from math import ceil

class Material:
    def __init__(self, file_path: str, simulation_days :int):
        self.file_path = file_path
        self.docs = self._load_pdf()
        self.simulation_days = simulation_days
        self.dayIndex = self._index_list()
        

    def __len__(self):
        return len(self.docs)
    
    def _load_pdf(self) -> list[Document]:
        loader = PyPDFLoader(self.file_path)
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        return docs
    
    def _index_list(self) -> list[int]:
        index = [0] + [ceil(i*(len(self.docs)/self.simulation_days)) for i in range(1, self.simulation_days)]
        index.append(len(self.docs)-1)
        return index
    
    def get_docs(self, accumulated_days :int, today :int, return_page: bool = False) -> list[Document] | list[int]:
        since = today - accumulated_days - 1
        start_page = max(0,self.dayIndex[since]-1)
        end_page = self.dayIndex[today]
        if return_page:
            return start_page, end_page, self.docs[start_page:end_page]
        return self.docs[start_page:end_page]
    
    def get_week_docs(self, day :int, return_page: bool = False) -> list[Document] | list[int]:
        start_page = self.dayIndex[day-1]
        end_page = self.dayIndex[day+5]
        if return_page:
            return start_page, end_page, self.docs[start_page:end_page]
        return self.docs[start_page:end_page]
