import re
import json

def parse_log(log_content):
    def day_as_key(data):
        result = {}
        for entry in data:
            day = entry['day']
            value = {key: entry[key] for key in entry if key != 'day'}
            if day not in result:
                result[day] = []
            result[day].append(value)
        return result
    
    prompt_pattern = r'PROMPT:\s*({.*?})\n'
    action_pattern = r'ACTION:\s*({.*?})\n'

    prompts = re.findall(prompt_pattern, log_content, re.DOTALL)
    actions = re.findall(action_pattern, log_content, re.DOTALL)
    actions = [action+'}' for action in actions]

    prompt_dicts = [json.loads(prompt) for prompt in prompts]
    action_dicts = [json.loads(action) for action in actions]
    
    return day_as_key(prompt_dicts), day_as_key(action_dicts)