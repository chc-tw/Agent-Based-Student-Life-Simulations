from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from dataclasses import dataclass
from src.memory import Memory
from src.material import Material
from src.util import WEEKDAY
from src.logger import Logger
from tenacity import retry, stop_after_attempt
import numpy as np
import json

@dataclass
class status:
    mood: int
    energy: int
    friends: int
    health: int
    learning_ability: int
    

class StudentAgent:
    study_plan = None
    def __init__(self, name : str, personality : str, config : dict, instructions : dict):
        self.name = name
        self.instructions = instructions
        self.personality = personality
        self.memory_config = config['Memory']
        self.llm_config = config['Agent']
        self.status_config = config['Status']
        self.llm = self._init_llm()
        self.memory = self._init_memory()
        self.material = self._init_material(config['System']['PDF_PATH'], config['System']['DAYS'])

        self.history = []
        self.status = status(100, 100, 0, 100, 100)
        self.max_token = self.llm_config['MAX_TOKEN']
        self.accumulated_materials = 0
        self.weekly_study_plan = ""
        self.action_dict = {
            "study": self.study,
            "relax": self.relax,
            "sleep": self.sleep,
            "socialize": self.socialize,
            "exercise": self.exercise
        }
        self.study_plan = None
        self.sick = False
        # Initialize logger for this agent
        self.logger = Logger(name, personality)

    def decideAction(self, day: int):
        input_prompt = """
        <current_status>
        {current_status}
        </current_status>

        <activites history>
        {history}
        </activites history>

        <accumulated materials>
        Number of accumulated materials: {accumulated_materials}
        </accumulated materials>

        Today is {day} {available_events}
        {sick_message}
        Your decision:
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instructions['DECIDE_INSTRUCTION']),
            ("human", input_prompt)
        ])
        
        inputs = {
            "current_status": str(self.status),
            "history": '\n'.join(self.history),
            "accumulated_materials": self.accumulated_materials,
            "day": day,
            "available_events": ", You can take course today." if day == "Monday" else "",
            "sick_message": ", However, you are sick today, so you cannot take course. You can choose to relax, sleep, socialize, or exercise." if self.sick else ""
        }
        
        chain = prompt | self.llm | StrOutputParser()
        output = chain.invoke(inputs)
        
        self.logger.log_prompt("decide_action_chain", prompt.format(**inputs), output)
        return output

    def study(self): 
        self.status.mood -= self.status_config['loss_mood_study']
        self.status.energy -= self.status_config['loss_energy_study']
        input_prompt = """
        <material>
        {material}
        </material>

        <study plan>
        {study_plan}
        </study plan>

        Your summary:
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instructions['STUDY_INSTRUCTION']),
            ("human", input_prompt)
        ])

        material = self.material.get_docs(self.accumulated_materials, len(self.history)+1)
        token_limit = self.max_token * (self.status.learning_ability/100)
        
        inputs = {
            "token_limit": token_limit,
            "material": material,
            "study_plan": self.weekly_study_plan
        }
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke(inputs)
        inputs['pages'] = self.material.get_docs(self.accumulated_materials, len(self.history)+1, return_page=True)
        self.logger.log_prompt("study", prompt.format(**inputs), summary)
        self.memory.memorize([summary])

    def answer_question(self, query: str):
        return self.memory.recall(query)
        
    def relax(self):
        self.status.mood += self.status_config['add_mood_relax']
    
    def sleep(self):
        self.status.energy += self.status_config['add_energy_sleep']

    def socialize(self):
        self.status.friends += 1
        self.status.mood += self.status_config['add_mood_socializing']
        self.status.energy = max(0, self.status.energy - self.status_config['loss_energy_socializing'])

    def exercise(self):
        self.status.health += self.status_config['add_health_exercise']

    def take_course(self):
        self.weekly_study_plan = self.study_plan
        self.status.mood -= self.status_config['loss_mood_take_courses']
        self.status.energy -= self.status_config['loss_energy_take_courses']

    def weekend(self):
        self.weekly_study_plan = ""
        self._sick()
        if self.sick:
            self.history.append(f"Weekend {(len(self.history)+1) // 7}: Get sick")
        self.memory.forget(self.accumulated_materials)
        return self.sick

    def _sick(self):
        if self.status.health < np.random.randint(1, 101):
            self.sick = True
        else:
            self.sick = False
    @retry(stop=stop_after_attempt(3))
    def takeAction(self, day: int):
        weekday = WEEKDAY[day%7]
        action = self.decideAction(weekday)
        try:
            self.action_dict[action]()
            self._update_max_token()
            if action == 'study':
                self.accumulated_materials = 0
            else:
                self.accumulated_materials += 1
            
            status_dict = self.status.__dict__
            self.history.append(f"Day{len(self.history)}: chose to {action}\n updated status: {self.status}")
            self.logger.log_action(action, status_dict, len(self.history))
            
        except KeyError:
            raise ValueError(f"Invalid action: {action}")
        finally:
            if action == 'take_course': # If the agent takes course, it can take other actions on the same day, and we use sick to prevent it from taking course again.
                self.sick = True
                self.takeAction(day)
            return action, self.status.__dict__

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
    
    def _update_max_token(self):
        self.status.learning_ability = max(10, (self.status.mood + self.status.energy)/2 * (self.status.friends * 10))
    
    def _init_memory(self):
        self.memory = Memory(self.memory_config, self.instructions, self.llm, self.name)
    
    @classmethod
    def update_study_plan(cls, self, day : int):
        material = cls.material.get_week_docs(day)
        start_page, end_page = cls.material.get_week_docs(day, return_page=True)
        student_input_prompt = """
        <material>
        {material}
        </material>

        <previous plan>
        {previous_plan}
        </previous plan>

        <feedback>
        {feedback}
        </feedback>
        """
        teacher_input_prompt = """
        <material>
        {material}
        </material>

        <student's study plan>
        {study_plan}
        </student's study plan>

        Your feedback:
        """
        
        student_prompt = ChatPromptTemplate.from_messages([
            ("system", cls.instructions['TAKE_COURSE_INSTRUCTION ']),
            ("human", student_input_prompt)
        ])
        
        teacher_prompt = ChatPromptTemplate.from_messages([
            ("system", cls.instructions['TEACH_INSTRUCTION']),
            ("human", teacher_input_prompt)
        ])
        
        plan = ""
        feedback = ""
        
        for _ in range(3): 
            # Student generates a study plan
            chain = student_prompt | self.llm | StrOutputParser()
            inputs = {"material": material,
                      "previous_plan": plan,
                      "feedback": feedback}
            plan = chain.invoke(inputs)
            inputs['pages'] = start_page, end_page
            self.logger.log_chain("take_course_student_chain", student_prompt.format(**inputs), plan)

            # Teacher provides feedback
            chain = teacher_prompt | self.llm | StrOutputParser()
            inputs = {"material": material,
                      "study_plan": plan}
            feedback = chain.invoke(inputs)
            inputs['pages'] = start_page, end_page
            self.logger.log_chain("take_course_teacher_chain", teacher_prompt.format(**inputs), feedback)

        # Student generates the final study plan
        chain = student_prompt | self.llm | StrOutputParser()
        inputs = {"material": material,
                  "previous_plan": plan,
                  "feedback": feedback}
        cls.study_plan = chain.invoke(inputs)

    def _init_material(cls, pdf_path : str, simulation_days : int):
        return Material(pdf_path, simulation_days)
    
    def __str__(self):
        return f"StudentAgent(name={self.name}, personality={self.personality}, status={self.status}, history={self.history})"
    
