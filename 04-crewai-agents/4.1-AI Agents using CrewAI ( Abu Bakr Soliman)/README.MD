# CrewAI

## Agent & Crew

*What Is an AI Agent?*  

- An **AI agent** is a system or program designed to autonomously perform tasks on behalf نيابةً of a 
  user or another system. It perceives its environment through inputs, takes actions based on its 
  reasoning and planning capabilities, and works to achieve predefined goals
- Human Delegate يُوَكل Agent to do some thing
   - **Autonomy**: الاستقلاليه Agents operate with a degree of independence, deciding actions without 
   continuous human prompts     

*What Is an AI Crew?*  

- An **AI Crew** is a structured, multi-agent system where each individual agents with a defined 
  role collaborate to perform a complex tasks that single agents cannot handle alone
   - **Role-Based Agents** : Each agent in the crew has a specific function
	 - **Collaborative Workflows**: Agents share intermediate results, delegate sub-tasks, and   
     iteratively refine outputs based on peer feedback
	 - **Tool and API Integration**: Crews can use external tools—databases, ML models, web services

## Native skills or External Tools ?  

- **Native Skills** (internal skills):  
	- These are the built-in abilities that an AI agent already has (Built-in knowledge)
		- Example: An AI agent that can read text, summarize it, and write responses  all using its own programming or model.
		- No extra tools needed.
		- Faster, but sometimes limited in what it can do.
- **External Help** (tool use or API integration)
	- AI agent uses outside tools or services to do tasks (Asking other tools to help)
		- Example: An AI agent that calls Google Translate to Translate or Use calendar to host a meeting.
		- More powerful and flexible, but sometimes slower or needs internet access.   

## Sequential Flow VS Hierarchical Flow
[DOC](https://docs.crewai.com/concepts/processes)
- Sequential Flow : **Agents work one after another**, passing results
	- A → B → C → D
- Hierarchical Flow : **One "manager agent" controls or coordinates other agents**, giving them tasks and combining results.
	- A → B,C || B → F ||
	- Ensure to provide a manager_llm or manager_agent
