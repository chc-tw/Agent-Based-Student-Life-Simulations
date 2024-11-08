from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import json



class TakeCourse:
  def __init__(self, materials):
    self.materials=json.load(open(materials))
    
    
