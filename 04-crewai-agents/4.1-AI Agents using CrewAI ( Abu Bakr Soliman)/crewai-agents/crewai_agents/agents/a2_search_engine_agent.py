from crewai import Agent
from crewai.tools import tool

from tavily import TavilyClient

from crewai_agents.config import llm
from crewai_agents.utilis import get_tavily_api_key

tavily_api_key = get_tavily_api_key()
tavily_client = TavilyClient(tavily_api_key)


@tool # Decorator indicating this function interacts with an external tool (Tavily)
def search_engine_tool(query: str):
    """Useful for search-based queries. Use this to find current information about any query related pages using a search engine"""
    print(f"[DEBUG] Searching with query: {query}")
    return tavily_client.search(query)

search_engine_agent = Agent(
    role="Search Engine Agent",
        goal=(
        "You are a web search expert.  \n"
        "When you need to look up a product, call the tool **search_engine_tool**.  \n"
        "Format your tool call exactly as:\n\n"
        "Action: search_engine_tool\n"
        "Action Input: {\"query\": \"<your search phrase here>\"}\n\n"
        "Then wait for the Observation before proceeding."
    ),
    # goal="To search for products based on the suggested search query",
    backstory="The agent is designed to help in looking for products by searching for products based on the suggested search queries.",
    llm=llm,
    verbose=True,
    tools=[search_engine_tool]   # New
)