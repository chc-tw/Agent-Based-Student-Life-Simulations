import yaml
from src.agents import StudentAgent
from src.util import WEEKDAY
from dotenv import load_dotenv
import os
import json
import asyncio
load_dotenv()

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('configs/Instructions.yaml', 'r') as file:
    instructions = yaml.safe_load(file)

with open('configs/agents.yaml', 'r') as file:
    agents_settings = yaml.safe_load(file)


def main():
    agents = []
    os.makedirs('./db', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    for agent_name, agent_personalities in agents_settings.items():
        agent = StudentAgent(name=agent_name,
                             personality=agent_personalities,
                             config=config,
                             instructions=instructions)
        agents.append(agent)
    simulation_days = config['System']['DAYS']
    history_data = {}

    for day in range(1, simulation_days+1):
        print(f"###Day {day} {WEEKDAY[day%7]} ###")
        for agent in agents:
            action, status = agent.takeAction(day)
            if day not in history_data:
                history_data[day] = {}
            history_data[day][agent.name] = [action, status]
            print(f"\t{agent.name} : {agent.history}")

            if day%7 == 0:
                for agent in agents:
                    agent.update_max_token()

    with open('agents_history.json', 'w') as json_file:
        json.dump(history_data, json_file, indent=4)


if __name__ == "__main__":
    main()