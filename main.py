import yaml
from src.agents import Agent
from dotenv import load_dotenv
import os
import json
load_dotenv()

with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('configs/Instruction.yaml', 'r') as file:
    instructions = yaml.safe_load(file)

with open('configs/agents.yaml', 'r') as file:
    agents_settings = yaml.safe_load(file)


def main():
    agents = []
    os.makedirs('./db', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    for agent_name, agent_personalities in agents_settings.items():
        agent = Agent(agent_name, agent_personalities, instructions, config)
        agents.append(agent)
    simulation_days = config['SYSTEM']['DAYS']
    history_data = {}

    for day in range(simulation_days):
        print(f"###Day {day + 1} ###")
        for agent in agents:
            agent.takeAction()
            if day not in history_data:
                history_data[day] = {}
            history_data[day][agent.name] = agent.history
            print(f"{agent.name} : {agent.history}")



    with open('agents_history.json', 'w') as json_file:
        json.dump(history_data, json_file, indent=4)


if __name__ == "__main__":
    main()