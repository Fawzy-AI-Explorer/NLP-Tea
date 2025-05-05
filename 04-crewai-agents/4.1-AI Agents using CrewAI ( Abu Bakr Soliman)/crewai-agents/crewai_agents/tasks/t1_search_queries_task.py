from crewai import Task
from pydantic import BaseModel, Field
from typing import List
import json
import os
from crewai_agents.agents.a1_search_queries_agent import search_queries_recommendation_agent
from crewai_agents.config import output_dir

# no_keywords=10
class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(..., title="Suggested search queries to be passed to the search engine",
                               min_items=1, max_items=3)

search_queries_recommendation_task = Task(
    description="\n".join([
        "Rankyx is looking to buy {product_name} at the best prices (value for a price strategy)",
        "The campany target any of these websites to buy from: {websites_list}",
        "The company wants to reach all available proucts on the internet to be compared later in another stage.",
        "The stores must sell the product in {country_name}",
        "Generate at maximum {no_keywords} queries.",
        "The search keywords must be in {language} language.",
        "Search keywords must contains specific brands, types or technologies. Avoid general keywords.",
        "The search query must reach an ecommerce webpage for product, and not a blog or listing page."
    ]),
    expected_output="A JSON object containing a list of suggested search queries.",
    output_json=SuggestedSearchQueries,
    output_file=os.path.join(output_dir, "step_1_suggested_search_queries.json"),
    agent=search_queries_recommendation_agent
)

# once it finishes, you get a Pydantic object SuggestedSearchQueries(queries=[…])
# CrewAI will automatically make that available to 
# Task #2 under the name of the field—here, queries.
# Task #2 can refer to {queries} in its own prompt   # IMPORTANT IMPORTANT IMPORTANT IMPORTANT

# In sequential mode, CrewAI will automatically merge the fields of Task #1’s output_json model into the next task’s context