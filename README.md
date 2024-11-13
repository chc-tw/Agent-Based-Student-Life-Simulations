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
We simulate the student life in a university environment. The whole experiemnt will last 8 weeks, just like a half semester. Every day, the agent will receive a material, and consider the history of activities and current status to choose one of the following activities to do:
- **Study**: The agents can choose to reduce mood and energy to study their daily material. If they don’t study, the material will accumulate, and the next time they choose to study, they have to study all the accumulated materials at once, leading to a bad study performance.
- **Take Course**: (Monday Only) The agents can choose to reduce mood and energy to take a course. If they do so, they will obtain the guidance of material this week from the teacher.
- **Socialize**: The agents can choose to reduce energy to add friends and increase their mood.
- **Relax**: The agents can choose to rest to increase their mood.
- **Sleep**: The agents can choose to sleep to increase their energy.
- **Exercise**: The agents can choose to exercise to increase their health.

#### Status Interaction
- Days since last time study: At the end of each week, the agent would forget some of their memories (the knowledge they learned) based on this formula: $100\times \text{forgetting factor}^{\text{days after last study}}\%$. The default forgetting factor is 0.9.
- Health: The agent will have a probability to get sick based on this formula: $\text{health}\%$. Once the agent gets sick, the agent won't be able to attend the course on Monday.
- Number of Friends: The agent can can have better study ability if they have more friends. One friend provides 10% more study ability.

After whole simulation, agents will have a final exam to validate their study result.

## Way to Simulate Student's Behavior
- **Learning Simulation**: We ask LLM to do the summarization of the material to simulate the student's learning.
- **Taking Course**: We use a teacher LLM (a larger model) to teach the student by providing the guidance of the material. Moreover, student can ask some feedbacks from the teacher.
- **Learning Ability**: We use token limitation in each prompt to limit the student's learning ability. The more the token is, the more detailed information the student can learn.
- **Memory and Forget**: We use a vector database to store the student's memories. Once the student forgets something, we remove the vector from the database randomly.
- **Take Exam**: We use RAG to simulate student's memory and answer the exam.

# Run Simulation
## Environment Setup
You can build the environment by running following command:
```bash
pipenv install
```

## Run the simulation
You can re-run the simulation by running following command (In our case, we use textbook of "Operating System Concepts by Avi Silberschatz, Peter Baer Galvin, and Greg Gagne" as the learning material):
```bash
pipenv run python main.py
```
## Customize the simulation
You can customize the simulation by changing the config in `configs/config.yaml`.

# Future Works
- [x] Enable using local model to perform simulation
- [ ] Enable customizing the simulation environment, including:
  - [ ] Learning textbooks & Exam questions
  - [ ] Agents' personalities
  - [ ] Simulation time
- [ ] Build a web app to visualize the simulation result
- [ ] Add more attributes for the agents to have.
- [ ] Add more activities for the agents to do.
