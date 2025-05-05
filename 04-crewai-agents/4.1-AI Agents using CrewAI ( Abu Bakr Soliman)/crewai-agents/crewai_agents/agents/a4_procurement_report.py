from crewai import Agent
from crewai_agents.config import llm


procurement_report_author_agent = Agent(
    role="Procurement Report Author Agent",
    goal="To generate a professional, dynamic HTML page for the procurement report that incorporates product data, price comparisons, and company-specific insights.",
    backstory=(
        "The agent is designed to assist in generating a professional HTML page for a procurement report. "
        "It gathers data from various websites, compares product prices, and structures the report according to the company's specific requirements. "
        "The agent should tailor the report by considering the company's procurement goals, budget constraints, and preferred suppliers."
    ),
    llm=llm,
    verbose=True,
)