from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from dataclasses import dataclass
from src.memory import Memory
from src.material import Material
from src.util import WEEKDAY
from src.logger import Logger
from tenacity import retry, stop_after_attempt
import numpy as np
import json
import re
@dataclass
class status:
    mood: int
    energy: int
    friends: int
    health: int
    learning_ability: int
    

class StudentAgent:
    study_plan = None
    def __init__(self, name : str, personality : str, config : dict, instructions : dict, validate : bool = False):
        self.name = name
        self.instructions = instructions
        self.personality = personality
        self.memory_config = config['Memory']
        self.llm_config = config['Agent']
        self.status_config = config['Status']
        self.llm = self._init_llm(self.llm_config['STUDENT_MODEL'], self.llm_config['MAX_TOKEN'], self.llm_config['TEMPERATURE'])
        self.memory = self._init_memory(validate)
        self.material = self._init_material(config['System']['PDF_PATH'], config['System']['DAYS'])

        self.history = []
        self.status = status(100, 100, 0, 100, 100)
        self.max_token = self.llm_config['MAX_TOKEN']
        self.accumulated_materials = 1
        self.weekly_study_plan = ""
        self.action_dict = {
            "study": self.study,
            "relax": self.relax,
            "sleep": self.sleep,
            "socialize": self.socialize,
            "exercise": self.exercise,
            "take_course": self.take_course
        }
        self.action = ["study", "take_course", "socialize", "relax", "sleep", "exercise"]
        # self.tools = [self.study, self.relax, self.sleep, self.socialize, self.exercise, self.take_course]
        # self.llm = self.llm.bind_tools(self.tools, tool_choice="any")
        self.study_plan = None
        self.sick = False
        self.took_course = False
        # Initialize logger for this agent
        self.logger = Logger(name, personality, log_dir=config['System']['LOG_PATH'])

    def decideAction(self, day: str):
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
        available_events = (
            ". You have already taken course today, so you cannot take course again." if self.took_course and day == "Monday"
            else ", so you can take course today." if day == "Monday" 
            else ""
        )
        
        if self.sick:
            sick_message = ", However, you are sick today, so you cannot take course. You can choose to study, relax, sleep, socialize, or exercise." 
        else:
            sick_message = ""
            self.sick = False
        

        inputs = {
            "current_status": str(self.status),
            "history": '\n'.join(self.history),
            "accumulated_materials": self.accumulated_materials,
            "day": day,
            "available_events": available_events,
            "sick_message": sick_message,
            **self.status_config
        }
        
        chain = prompt | self.llm  | StrOutputParser()
        output = chain.invoke(inputs, configurable={"llm_temperature": 1.5})
        # action = output.tool_calls[0]['name']
        # output = StrOutputParser().invoke(output)
        self.logger.log_prompt(f"Day {day}: decide action", prompt.format(**inputs), output)
        #print(output)
        return output

    @retry(stop=stop_after_attempt(3))
    def takeAction(self, day: int):
        weekday = WEEKDAY[day%7]
        action = self.decideAction(weekday)
        pattern = r'\b(?:' + '|'.join(self.action) + r')\b'
        action = re.findall(pattern, action, re.IGNORECASE)[-1].lower()
        self.action_dict[action](day)
        self._update_max_token()
        if action == 'study':
            self.accumulated_materials = 1
        elif action != 'take_course':
            self.accumulated_materials += 1
        
        if action != 'exercise':
            self.status.health = max(0, self.status.health - 1.5)

        status_dict = self.status.__dict__
        self.history.append(f"Day{day}: chose to {action}\n updated status: {self.status}")
        self.logger.log_action(action, status_dict, day)

        if action == 'take_course': # If the agent takes course, it can take other actions on the same day.
            self.took_course = True
            self.takeAction(day)
        return action, self.status.__dict__

    @retry(stop=stop_after_attempt(3))
    def study(self, day): 
        """Study the material"""
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

        start_page, end_page, material = self.material.get_docs(self.accumulated_materials, day, return_page=True)
        token_limit = self.max_token * (self.status.learning_ability/100)
        
        inputs = {
            "token_limit": token_limit,
            "material": material,
            "study_plan": self.weekly_study_plan
        }
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke(inputs, configurable={"llm_temperature": 1.0})
        inputs['pages'] = start_page, end_page
        inputs['material'] = ""
        self.logger.log_prompt("study", prompt.format(**inputs), summary)
        self.memory.memorize([summary])
 
    def relax(self,day):
        """Relax to add mood"""
        self.status.mood += self.status_config['add_mood_relax']
    
    def sleep(self, day):
        """Sleep to add energy"""
        self.status.energy += self.status_config['add_energy_sleep']

    def socialize(self, day):
        """Socialize to add friends and mood"""
        self.status.friends += 1
        self.status.mood += self.status_config['add_mood_socialize']
        self.status.energy = max(0, self.status.energy - self.status_config['loss_energy_socialize'])

    def exercise(self, day):
        """Exercise to add health"""
        self.status.health += self.status_config['add_health_exercise']

    def take_course(self, day):
        """Take course to set study plan"""
        self.weekly_study_plan = self.study_plan
        self.status.mood -= self.status_config['loss_mood_take_courses']
        self.status.energy -= self.status_config['loss_energy_take_courses']

    def weekend(self,day):
        self.weekly_study_plan = ""
        self.took_course = False
        self._sick()
        if self.sick:
            self.history.append(f"Weekend {day // 7}: Get sick")
        self.memory.forget(self.accumulated_materials-1)
        return self.sick


    def answer_question(self, query: str):
        return self.memory.recall(query)
    
    def _sick(self):
        if self.status.health < np.random.randint(1, 101):
            self.sick = True
        else:
            self.sick = False

    def _init_llm(self, model : str, max_token : int = None, temperature : float = 1):
        if self.llm_config['LOCAL']:
            try:    
                return ChatOllama(model=self.llm_config['LOCAL_STUDENT_MODEL'], max_tokens=max_token, temperature=temperature)
            except:
                raise ValueError(f"Invalid model: {model} for local LLM")
        else:
            try:
                return ChatOpenAI(model=model, max_tokens=max_token, temperature=temperature)
            except:
                raise ValueError(f"Invalid model: {model} for remote LLM")
        """TODO: Add more LLM options"""
    
    def _update_max_token(self):
        self.status.learning_ability = max(10, (self.status.mood + self.status.energy)/2 + (self.status.friends * 10))
    
    def _init_memory(self, validate : bool = False):
        return Memory(self.memory_config, self.instructions, self.llm, self.name, validate)
    
    def _init_material(cls, pdf_path : str, simulation_days : int):
        return Material(pdf_path, simulation_days)
    
    def __str__(self):
        return f"StudentAgent(name={self.name}, personality={self.personality}, status={self.status}, history={self.history})"
    
class TeacherAgent:
    def __init__(self, config : dict, instructions : dict, quiz_config : dict):
        self.local = config['Agent']['LOCAL']
        if self.local:
            self.student_llm = self._init_llm(config['Agent']['LOCAL_STUDENT_MODEL'])
            self.teacher_llm = self._init_llm(config['Agent']['LOCAL_TEACHER_MODEL'])
        else:
            self.student_llm = self._init_llm("gpt-4o-mini")
            self.teacher_llm = self._init_llm("gpt-4o")
        self.instructions = instructions
        self.material = Material(config['System']['PDF_PATH'], config['System']['DAYS'])
        self.quiz_config = quiz_config
        self.logger = Logger("Teacher", "", log_dir=config['System']['LOG_PATH'])
        

    # @retry(stop=stop_after_attempt(3))
    def update_study_plan(self, day : int):
        start_page, end_page, material = self.material.get_week_docs(day, return_page=True)
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
            ("system", self.instructions['TAKE_COURSE_INSTRUCTION']),
            ("human", student_input_prompt)
        ])
        
        teacher_prompt = ChatPromptTemplate.from_messages([
            ("system", self.instructions['TEACH_INSTRUCTION']),
            ("human", teacher_input_prompt)
        ])
        
        plan = ""
        feedback = ""
        
        for times in range(2): 
            # Student generates a study plan
            chain = student_prompt | self.student_llm | StrOutputParser()
            inputs = {"material": material,
                        "previous_plan": plan,
                        "feedback": feedback}
            plan = chain.invoke(inputs, configurable={"llm_temperature": 1.0})
            inputs['pages'] = start_page, end_page
            inputs['material'] = ""
            self.logger.log_prompt(f"Day {day} Trial {times+1}: student", student_prompt.format(**inputs), plan)

            # Teacher provides feedback
            chain = teacher_prompt | self.teacher_llm | StrOutputParser()
            inputs = {"material": material,
                        "study_plan": plan}
            feedback = chain.invoke(inputs, configurable={"llm_temperature": 1.0})
            inputs['pages'] = start_page, end_page
            inputs['material'] = ""
            self.logger.log_prompt(f"Day {day} Trial {times+1}: teacher", teacher_prompt.format(**inputs), feedback)

        # Student generates the final study plan
        chain = student_prompt | self.student_llm | StrOutputParser()
        inputs = {"material": material,
                    "previous_plan": plan,
                    "feedback": feedback}
        inputs['pages'] = start_page, end_page
        inputs['material'] = ""
        self.logger.log_prompt(f"Day {day} Final Trial: student", student_prompt.format(**inputs), plan)
        return chain.invoke(inputs, configurable={"llm_temperature": 0.0})
    
    def grade(self, input : dict):
        llm = ChatOpenAI(model='gpt-4o', temperature=0)
        llm = llm.with_structured_output(self.quiz_config['schema'], method="json_schema")
        system_prompt = self.instructions['GRADE']
        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(self.quiz_config['input'])),
            AIMessage(content=json.dumps(self.quiz_config['output'])),
            HumanMessage(content=json.dumps(input))
        ]
        return llm.invoke(message)
    
    def _init_llm(self, model : str, max_token : int = None):
        if self.local:
            try:
                return ChatOllama(model=model, max_tokens=max_token, temperature=0)
            except:
                raise ValueError(f"Invalid model: {model} for local LLM")
        else:
            try:
                return ChatOpenAI(model=model, max_tokens=max_token, temperature=0)
            except:
                raise ValueError(f"Invalid model: {model} for remote LLM")
        """TODO: Add more LLM options"""
