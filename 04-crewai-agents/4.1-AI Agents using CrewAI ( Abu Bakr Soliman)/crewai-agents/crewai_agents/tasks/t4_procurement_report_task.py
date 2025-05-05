from crewai import Task
import os
from crewai_agents.agents.a4_procurement_report import procurement_report_author_agent
from crewai_agents.config import output_dir



procurement_report_author_task = Task(
    description="\n".join([
        "The task is to generate a professional HTML page for the procurement report with the following structure:",
        "1. Executive Summary: A brief overview of the procurement process and key findings.",
        "2. Introduction: An introduction to the purpose and scope of the report, including company-specific insights.",
        "3. Methodology: A detailed description of the methods used to gather and compare prices from different sources.",
        "4. Findings: A dynamic table displaying product data (title, price, capacity, material) at least 5 products sourced from multiple websites.",
        "5. Analysis: In-depth analysis of the findings, highlighting significant trends, price discrepancies, and recommendations for suppliers.",
        "6. Recommendations: Actionable procurement recommendations based on the analysis, including potential supplier choices.",
        "7. Conclusion: A concise summary of the report with key takeaways and next steps.",
        "8. Appendices: Any supplementary data, charts, or raw product data."
    ]),

    expected_output="A professional, fully formatted HTML procurement report with dynamic content based on provided product data.",
    output_file=os.path.join(output_dir, "step_4_procurement_report.html"),
    agent=procurement_report_author_agent,
)