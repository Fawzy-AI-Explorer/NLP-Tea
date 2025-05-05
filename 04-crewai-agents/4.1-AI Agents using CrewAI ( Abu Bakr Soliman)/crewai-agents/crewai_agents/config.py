from crewai import LLM


# set the output directory for the agent
output_dir= r"outputs/ai-agent-output"


# Initialize LLM
llm = LLM(
    model="ollama/deepseek-r1",
    base_url="http://localhost:11434", 
    temperature=0.5
)


