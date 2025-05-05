from crewai import Agent
from crewai.tools import tool
from scrapegraph_py import Client
from pydantic import BaseModel, Field
from typing import List

from crewai_agents.config import llm
from crewai_agents.utilis import get_scrap_api_key


scrap_key = get_scrap_api_key()
scrap_client = Client(api_key=scrap_key)





class ProductSpec(BaseModel):
    specification_name: str
    specification_value: str

class SingleExtractedProduct(BaseModel):
    page_url: str = Field(..., title="The original url of the product page")
    product_title: str = Field(..., title="The title of the product")
    product_image_url: str = Field(..., title="The url of the product image")
    product_url: str = Field(..., title="The url of the product")
    product_current_price: float = Field(..., title="The current price of the product")
    product_original_price: float = Field(title="The original price of the product before discount. Set to None if no discount", default=None)
    product_discount_percentage: float = Field(title="The discount percentage of the product. Set to None if no discount", default=None)

    product_specs: List[ProductSpec] = Field(..., title="The specifications of the product. Focus on the most important specs to compare.", min_items=1, max_items=5)

    agent_recommendation_rank: int = Field(..., title="The rank of the product to be considered in the final procurement report. (out of 5, Higher is Better) in the recommendation list ordering from the best to the worst")
    agent_recommendation_notes: List[str]  = Field(..., title="A set of notes why would you recommend or not recommend this product to the company, compared to other products.")


class AllExtractedProducts(BaseModel):
    products: List[SingleExtractedProduct]

@tool
def web_scraping_tool(page_url: str):
    """
    An AI Tool to help an agent to scrape a web page

    Example:
    web_scraping_tool(
        page_url="https://www.noon.com/egypt-en/15-bar-fully-automatic-espresso-machine-1-8-l-1500"
    )
    """
    details = scrap_client.smartscraper(
        website_url=page_url,
        user_prompt="Extract ```json\n" + SingleExtractedProduct.schema_json() + "```\n From the web page"
    )

    return {
        "page_url": page_url,
        "details": details
    }


scraping_agent = Agent(
    role="Web scraping agent",
    # goal="To extract details from any website",
    goal="\n".join([
      "To extract details from any website",
      "When you return your final JSON, it MUST use the top‑level key `products` (plural).",
      "Example:",
      "Final Answer:",
      "{",
      '  "products": [ { … }, { … } ]',
      "}"
    ]),
    backstory="The agent is designed to help in looking for required values from any website url. These details will be used to decide which best product to buy.",
    llm=llm,
    tools=[web_scraping_tool],
    verbose=True,
)