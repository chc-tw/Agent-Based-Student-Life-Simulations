import yaml
from src.agents import StudentAgent, TeacherAgent
from src.util import WEEKDAY, calculate_accuracy_rate
from dotenv import load_dotenv
import os
import json
import asyncio
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import logging
load_dotenv()
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('configs/Instructions.yaml', 'r') as file:
    instructions = yaml.safe_load(file)

with open('configs/agents.yaml', 'r') as file:
    agents_settings = yaml.safe_load(file)
with open('configs/quiz_config.json', 'r') as file:
    quiz_config = json.load(file)
with open('exam2.json', 'r') as file:
    exam = json.load(file)

def main(grade : bool = False):
    agents = []
    log_dir = config['System']['LOG_PATH'] = config['System']['LOG_PATH'].format(date_time=datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs('./db', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    teacher = TeacherAgent(config=config, instructions=instructions, quiz_config=quiz_config)
    for agent_name, agent_personalities in agents_settings.items():
        agent = StudentAgent(name=agent_name,
                             personality=agent_personalities,
                             config=config,
                             instructions=instructions,
                             validate=False)
        agents.append(agent)
    simulation_days = config['System']['DAYS']
    history_data = {}

    for day in range(1, simulation_days+1):
        print(f"###Day {day} {WEEKDAY[day%7]} ###")
        if day%7 == 1:
            print("Teacher is updating study plan...")
            study_plan = teacher.update_study_plan(day)
            StudentAgent.study_plan = study_plan
        for agent in agents:
            action, status = agent.takeAction(day)
            if day not in history_data:
                history_data[day] = {}
            history_data[day][agent.name] = [action, status]
            print(f"\t{agent.name} : {agent.history[-1]}")

        if day%7 == 0:
            for agent in agents:
                sick = agent.weekend(day)
                if sick:
                    print(f"\t{agent.name} : Get sick")
                    history_data[day][agent.name].append("Get sick")

    with open(f'{log_dir}/agents_history.json', 'w') as json_file:
        json.dump(history_data, json_file, indent=4)

    for quiz_id, quiz in tqdm(exam.items(), desc="Answering quizzes", unit="quiz"):
        question = quiz['question']
        reply = []
        for agent in agents:
            reply.append(agent.answer_question(question))
            exam[quiz_id]['students'] = reply

    

    if not grade:
        return
    ## Grade the quizzes
    grade = {}
    batch = {}

    for quiz_id, quiz in tqdm(exam.items(), desc="Grading quizzes", unit="quiz"):
        batch.update({quiz_id: quiz})
        if len(batch) == 5:
            grade.update(teacher.grade(batch))
            batch = {}
    if len(batch) > 0:
        grade.update(teacher.grade(batch))

    accuracy_rate = calculate_accuracy_rate(grade)
    for student, rate in accuracy_rate.items():
        stdent_name = agents[student].name
        print(f"{stdent_name} : {rate}")

    
    with open(f'{log_dir}/agents_grade.json', 'w') as json_file:
        json.dump(grade, json_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grade", help="Grade the quizzes or not", default=True, action="store_true")
    args = parser.parse_args()
    main(args.grade)
