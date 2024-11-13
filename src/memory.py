from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
import os
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

class Memory:
    def __init__(self, config : dict, instructions : dict, llm, namespace : str):
        self.config = config
        self.namespace = namespace
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['OVERLAP'],
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        self.vector_store = self._init_vector_store()
        self.llm = llm
        rag_template = instructions['ANSWER_QUESTION_INSTRUCTION']
        self.rag_prompt = PromptTemplate.from_template(rag_template)
        
    def memorize(self, texts : List[str]):
        docs = self.text_splitter.create_documents(texts)
        if self.config['LOCAL']:
            self.vector_store.add_documents(docs, self.embedding)
        else:
            self.vector_store.add_documents(docs, self.embeddings, namespace=self.namespace)
    
    def recall(self, query : str):
        if self.config['LOCAL']:
            rag_chain = (
                {"context": self.vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()} 
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
            )
        else:
            rag_chain = (
                {"context": self.vector_store.as_retriever(search_kwargs={ "namespace": self.namespace }) | format_docs, "question": RunnablePassthrough()} 
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
            )
        return rag_chain.invoke({"question": query})
    
    def _init_vector_store(self):
        if self.config['LOCAL']:
            self.vector_store = Chroma(embedding_function=self.embeddings,
                persist_directory="./db",
                collection_name=self.namespace
            )
        else:
            self.vector_store = PineconeVectorStore(
                index_name=os.environ.get("PINECONE_INDEX_NAME"),
                embedding=self.embeddings,
                namespace=self.namespace
            )
        return self.vector_store

    def _init_embeddings(self):
        if self.config['LOCAL']:
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = OllamaEmbeddings(model=self.config['EMBEDDING_MODEL'])
        return self.embeddings