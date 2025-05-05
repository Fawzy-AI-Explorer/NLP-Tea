from crewai_agents.agents.a1_search_queries_agent import search_queries_recommendation_agent
from crewai_agents.tasks.t1_search_queries_task import search_queries_recommendation_task

from crewai_agents.agents.a2_search_engine_agent import search_engine_agent
from crewai_agents.tasks.t2_search_engine_task import search_engine_task

from crewai_agents.agents.a3_scraping_agent import scraping_agent
from crewai_agents.tasks.t3_scraping_task import scraping_task

from crewai_agents.agents.a4_procurement_report import procurement_report_author_agent
from crewai_agents.tasks.t4_procurement_report_task import procurement_report_author_task

from crewai_agents.utilis import get_agentops_api_key, set_agentops_api_key 

from crewai import Crew, Process



def run_search_engine_agent():
    """Run the search engine agent and return the results."""
    print("Running search engine agent...")
    # Set the AgentOps API key
    api_key = get_agentops_api_key()
    set_agentops_api_key(api_key)
    print("AgentOps API key set successfully.")

    crew = Crew(
        agents=[
            search_queries_recommendation_agent,
            search_engine_agent,
            scraping_agent,
            procurement_report_author_agent
              ],

        tasks=[
            search_queries_recommendation_task,
            search_engine_task,
            scraping_task,
            procurement_report_author_task          
            ],
        verbose=True,
        process=Process.sequential
    )
    print("Crew initialized successfully.")
        
    results = crew.kickoff(
        inputs={                
        "product_name": "book for professional development",
        "websites_list": ["amazon.eg", "jumia.com.eg", "noon.com"],
        "country_name": "Egypt",
        "no_keywords": 3,
        "language":"english",
        "score_th":0.1,
        "top_recommendations_no": 5,
        } 
    )
    print("Crew kickoff completed successfully.")
    return results 

if __name__ == "__main__":
    results = run_search_engine_agent()
    print("Search queries recommendation task completed successfully.")
    print(f"Results: {results}")


# To Run This Script:
#           cd E:\DATA SCIENCE\projects\crewai-agents22>
#           python -m examples.ex3_run_procurement_report_agent



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