import os

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()


def get_agentops_api_key() -> str:
    """
    Returns AgentOps API key from the environment.
    """
    key = os.getenv("AGENTOPS_API_KEY")
    if not key:
        raise RuntimeError("AGENTOPS_API_KEY not found in environment")
    return key

def set_agentops_api_key(api_key):
    """
    Sets AgentOps API key as an environment variable.
    """
    os.environ["AGENTOPS_API_KEY"] = api_key


def get_tavily_api_key() -> str:
    """
    Returns TAVILY_API_KEY from the environment.
    """
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not found in environment")
    return key

def get_scrap_api_key() -> str:
    """
    Returns SCRAP_API_KEY from the environment.
    """
    key = os.getenv("SCRAP_API_KEY")
    if not key:
        raise RuntimeError("SCRAP_API_KEY not found in environment")
    return key