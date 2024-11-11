from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from dataclasses import dataclass
from src.memory import Memory
from typing import Union
from tenacity import retry, stop_after_attempt

@dataclass
class status:
    Mood: int
    Energy: int
    friends: int
    health: int

class StudentAgent:
    def __init__(self, name : str, personality : str, config : dict, instructions : dict):
        self.memory_config = config['Memory']
        self.llm_config = config['Agent']
        self.status_config = config['Status']
        self.name = name
        self.llm = self._init_llm()
        self.memory = self._init_memory()
        self.personality = personality

        self.history = []
        self.status = status(100, 100, 0, 100)
        self.instructions = instructions
        self.max_token = self.llm_config['MAX_TOKEN']
        self.accumulated_materials = 0

        self.action_dict = {
            "study": self.study,
            "relax": self.relax,
            "sleep": self.sleep,
            "socialize": self.socialize,
            "exercise": self.exercise
        }

    def study(self, material: str): 
        self.status.Mood -= self.status_config['loss_mood_study']
        self.status.Energy -= self.status_config['loss_energy_study']
        self.history.append(f"Day{len(self.history)}: studied")
        input_prompt = """
        <material>
        {material}
        </material>
        Your summary:
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instructions['STUDY_INSTRUCTION']),
            ("user", input_prompt)
        ])
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"material": material, "token_limit": self.max_token})
        self.accumulated_materials = 0
        self.memory.memorize([summary])

    def answer_question(self, query: str):
        return self.memory.recall(query)
        
    def relax(self):
        self.status.Mood += self.status_config['add_mood_relax']
        self.history.append(f"Day{len(self.history)}: relaxed \n updated status: {self.status}")
        self.accumulated_materials += 1
    
    def sleep(self):
        self.status.Energy += self.status_config['add_energy_sleep']
        self.history.append(f"Day{len(self.history)}: slept \n updated status: {self.status}")
        self.accumulated_materials += 1

    def socialize(self):
        self.status.friends += 1
        self.status.Mood += self.status_config['add_mood_socializing']
        self.status.Energy -= self.status_config['loss_energy_socializing']
        self.history.append(f"Day{len(self.history)}: socialized \n updated status: {self.status}")
        self.accumulated_materials += 1

    def exercise(self):
        self.status.health += self.status_config['add_health_exercise']
        self.history.append(f"Day{len(self.history)}: exercised \n updated status: {self.status}")
        self.accumulated_materials += 1

    def decideAction(self):
        input_prompt = """
        <current_status>
        {current_status}
        </current_status>

        <activites history>
        {history}
        </activites history>
        <accumulated materials>
        {accumulated_materials}
        </accumulated materials>
        Your decision:
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instructions['DECISION_INSTRUCTION']),
            ("user", input_prompt)
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"current_status": self.status, "history": self.history, "accumulated_materials": self.accumulated_materials})
    
    @retry(stop=stop_after_attempt(3))
    def takeAction(self):
        action = self.decideAction()
        try:
            return self.action_dict[action]()
        except KeyError:
            raise ValueError(f"Invalid action: {action}")
    
    def _init_llm(self):
        if self.llm_config['LOCAL']:
            try:
                self.llm = ChatOllama(model=self.llm_config['MODEL'], max_tokens=self.llm_config['MAX_TOKEN'])
            except:
                raise ValueError(f"Invalid model: {self.llm_config['MODEL']} for local LLM")
        else:
            try:
                self.llm = ChatOpenAI(model=self.llm_config['MODEL'], max_tokens=self.llm_config['MAX_TOKEN'])
            except:
                raise ValueError(f"Invalid model: {self.llm_config['MODEL']} for remote LLM")
        """TODO: Add more LLM options"""
    
    def _init_memory(self):
        self.memory = Memory(self.memory_config, self.instructions, self.llm, self.name)

    def __str__(self):
        return f"StudentAgent(name={self.name}, personality={self.personality}, status={self.status}, history={self.history})"