from crewai import Crew, Process
from crewai_agents.agents.a1_search_queries_agent import search_queries_recommendation_agent
from crewai_agents.tasks.t1_search_queries_task import search_queries_recommendation_task
from crewai_agents.utilis import get_agentops_api_key, set_agentops_api_key 


def run_search_queries_agent():
    """Run the search queries recommendation agent and return the results."""

    api_key = get_agentops_api_key()
    set_agentops_api_key(api_key)

    crew = Crew(
        agents=[search_queries_recommendation_agent],
        tasks=[search_queries_recommendation_task],
        verbose=True,
        process=Process.sequential
    )

    results = crew.kickoff(
        inputs={                 # if the Task doesn't include any variables, you wouldn't need to include the inputs argument
        "product_name": "coffee machine for the office",
        "websites_list": ["amazon.eg", "jumia.com.eg", "noon.com"],
        "country_name": "Egypt",
        "no_keywords": 10,
        "language":"english"
        } 
     )
    return results 

def print_json():
    import json
    with open(r"E:\DATA SCIENCE\projects\crewai-agents22\outputs\ai-agent-output\step_1_suggested_search_queries.json") as f:
        data = json.load(f)
    print("type of data: ", type(data))   # <class 'dict'>
    print("type of data[queries]: ", type(data["queries"])) # <class 'list'>

    print(data["queries"], "\n")
    for q in data["queries"]:
        print(type(q), q) # <class 'str'> 
    

if __name__ == "__main__":
    # results = run_search_queries_agent()
    print("Search queries recommendation task completed successfully.")
    # print(f"Results: {results}") # Pydantic object
    print("*"*90, "\n")
    print_json()

# To Run This Script:
#           cd E:\DATA SCIENCE\projects\crewai-agents22>
#           python -m examples.ex1_run_search_queries_agent


'''
Task Execution Flow
1. prompt
    -  replaces each {…} with the value from inputs in Task description :
    -  prompt sent to your search_queries_recommendation_agent
2. Agent → LLM
    -  LLM generates a response based on the prompt
    -  LLM response is a almost string of JSON object
3. Validation (output_json=SuggestedSearchQueries)
4. save the output to a file (output_file=os.path.join(output_dir, "step_1_suggested_search_queries.json"))
      - dict of List of strings (queries) in JSON format

# once it finishes, you get a Pydantic object SuggestedSearchQueries(queries=[…])
# CrewAI will automatically make that available to 
# Task #2 under the name of the field—here, queries.
# Task #2 can refer to {queries} in its own prompt   # IMPORTANT

'''