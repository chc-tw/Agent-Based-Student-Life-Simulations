from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from collections import defaultdict

def fetch_pages(page_range  : tuple[int, int], docs : list[Document]) -> str:
    pages = []
    for page in range(page_range[0], page_range[1]):
        pages.append(docs[page].page_content)
    return '\n'.join(pages)


def generate_prompt(task_or_prompt: str) -> str:
    messages = [
        SystemMessage(content=META_PROMPT),
        HumanMessage(content="Task, Goal, or Current Prompt:\n" + task_or_prompt),
    ]
    llm = ChatOpenAI(model="gpt-4o")
    chain = llm | StrOutputParser()
    return chain.invoke(messages)

def generate_personality(personality_description: str) -> str:
    messages = [
        SystemMessage(content=PERSONALITY_PROMPT),
        HumanMessage(content="Personality Description:\n" + personality_description),
    ]
    llm = ChatOpenAI(model="gpt-4o")
    chain = llm | StrOutputParser()
    return chain.invoke(messages)
import json

def calculate_accuracy_rate(results: list[dict]) -> dict:
    accuracy_rates = defaultdict(int)
    total_count = len(results)

    for question_id, response in enumerate(results, start=1):
        for student, answer in enumerate(response[str(question_id)]):
            if answer == "correct":
                accuracy_rates[student] += 1

    return {student: count / total_count for student, count in accuracy_rates.items()}


WEEKDAY = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    0: "Sunday"
}
META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()

PERSONALITY_PROMPT = """
For the following experiment:
<experiment description>
# Agent-Based Student Life Simulations: Discovering the Balance Between Learning and Living
In this project, we explore the concept of agent-based simulations to model student life. The goal is to discover the balance between learning and living. We utilize the prompt engineering to simulate the student's learning properties and RAG (Retrieval Augmented Generation) to simulate the student's memory.

## Experiment Design
### Agent Settings
In our experiment, every agent has the 4 attributes:
- **Mood**: The mood of the agent. The value of mood will affect the agent's learning ability.
- **Energy**: The energy of the agent. The value of energy will affect the agent's learning ability.
- **Friends**: The number of friends the agent has. More friends means higher learning ability, just like we can learn from our friends.
- **Health**: The health of the agent. The lower the health is, the more likely the agent will get sick.

Besides, we design 4 different agents with different personalities and learning styles to experience different student life:
- **Sheldon**: Super Hardworking - thinks grades are everything and doesn’t want to waste time on other things unless they are necessary.
- **Nobita**: Super Lazy - prefers to be a sofa potato and doesn’t like books.
- **Capybara**: Values friendship above all - cannot stand loneliness, tries to maintain a work-life balance but struggles because he loves to hang out with friends.
- **Impossible**: Aims to balance everything in life - doesn’t want to be sick, tired, or depressed, but also wants to achieve good grades.
  
### Simulation Design
We simulate the student life in a university environment. The whole experiemnt will last 6 weeks, just like a half semester. Every day, the agent will receive a material, and consider the history of activities and current status to choose one of the following activities to do:
- **Study**: The agents can choose to reduce mood and energy to study their daily material. If they don’t study, the material will accumulate, and the next time they choose to study, they will have to cover all the accumulated materials at once.
- **Take Course**: (Monday Only) The agents can choose to reduce mood and energy to take a course. If they do so, they will obtain the guidance of material this week from the teacher.
- **Socialize**: The agents can choose to reduce energy to add friends and increase their mood.
- **Relax**: The agents can choose to rest to increase their mood.
- **Sleep**: The agents can choose to sleep to increase their health.
- **Exercise**: The agents can choose to exercise to increase their health.

#### Status Interaction
- Days since last time study: At the end of each week, the agent would forget some of their memories (the knowledge they learned) based on this formula: $100\times \text{forgetting factor}^{\text{days after last study}}%$. The default forgetting factor is 0.9.
- Health: The agent will have a probability to get sick based on this formula: $\text{health}%$. Once the agent gets sick, the agent won't be able to attend the course on Monday.
- number of friends: The agent can can have better study ability if they have more friends. One friend provides 10% more study ability.

After whole simulation, agents will have a final exam to test their study result.
</experiment description>
Based on the above experiment design, generate agent's personality description with users's high level personality description.
You should start with "You are" and end with "."
""".strip()