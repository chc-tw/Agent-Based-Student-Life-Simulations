from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
from math import ceil
from random import sample
import os
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

class Memory:
    def __init__(self, config : dict, instructions : dict, llm, namespace : str, validate : bool = False):
        self.config = config
        self.namespace = namespace
        self.validate = validate
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['CHUNK_SIZE'],
            chunk_overlap=config['OVERLAP'],
        )
        self.embedding = self._init_embeddings()
        self.vector_store = self._init_vector_store()
        self.llm = llm
        rag_template = instructions['ANSWER_QUESTION_INSTRUCTION']
        self.rag_prompt = PromptTemplate.from_template(rag_template)
        self.forget_factor = config['FORGET_FACTOR']
        self.memory_id = []
        
    def memorize(self, texts : List[str]):
        docs = self.text_splitter.create_documents(texts)
        if self.config['LOCAL']:
            self.memory_id += self.vector_store.add_documents(docs)
        else:
            self.memory_id += self.vector_store.add_documents(docs, namespace=self.namespace)
    
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
        return rag_chain.invoke(query)
    
    def forget(self, days_since_last_study : int):
        if self.memory_id:
            num_to_forget = ceil(len(self.memory_id) * (1 - self.forget_factor ** days_since_last_study))
            ids_to_forget = sample(self.memory_id, num_to_forget)
            if self.config['LOCAL']:
                self.vector_store.delete(ids_to_forget)
            else:
                self.vector_store.delete(ids_to_forget, namespace=self.namespace)
            self.memory_id = [id for id in self.memory_id if id not in ids_to_forget]

    def _init_vector_store(self):
        self.vector_store = PineconeVectorStore(
            index_name=os.environ.get("PINECONE_INDEX_NAME"),
            embedding=self.embeddings,
            namespace=self.namespace
        )
        return self.vector_store

    def _init_embeddings(self):
        if self.config['LOCAL']:
            self.embeddings = OllamaEmbeddings(model=self.config['LOCAL_EMBED'])
        else:
            self.embeddings = OpenAIEmbeddings()
        return self.embeddings
