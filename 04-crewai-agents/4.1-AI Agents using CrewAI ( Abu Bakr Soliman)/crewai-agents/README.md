# CrewAI Procurement Agents 🤖

A modular AI agent system built with CrewAI for product research, procurement, and analysis in e-commerce environments.

## Project Overview 🔍

This project implements a multi-agent system using CrewAI to automate the process of researching products, searching e-commerce websites, scraping relevant information, and generating procurement reports. The system is designed to help businesses make informed purchasing decisions by collecting and analyzing product data from various online sources.

## Features ✨

- **Search Query Generation**: AI agent that generates optimized search queries for product research
- **Search Engine Processing**: Agent that queries e-commerce sites and extracts relevant results
- **Web Scraping**: Agent that collects detailed product information from search results
- **Procurement Reports**: Agent that analyzes scraped data and creates comprehensive procurement reports

## Installation 💻

1. Clone the repository:
   ```
   git clone https://github.com/Fawzy-AI-Explorer/crewai-agents.git
   cd crewai-agents
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```
   # Create a .env file with your API keys
   OPENAI_API_KEY=your_openai_api_key
   AGENTOPS_API_KEY=your_agentops_api_key
   # Add other API keys as needed
   ```

## Project Structure 📂

```
crewai-agents/
│
│
├── crewai_agents/                   - Core module containing all agent definitions
│   ├── __init__.py                  - Package initialization
│   ├── config.py                    - Configuration settings
│   ├── utilis.py                    - Utility functions
│   │
│   ├── agents/                      - Individual agent implementations
│   │   ├── __init__.py
│   │   ├── a1_search_queries_agent.py   - Search query generation agent
│   │   ├── a2_search_engine_agent.py    - Search engine processing agent
│   │   ├── a3_scraping_agent.py         - Web scraping agent
│   │   └── a4_procurement_report.py     - Procurement report generation agent
│   │
│   │
│   └── tasks/                       - Task definitions for each agent
│       ├── __init__.py
│       ├── t1_search_queries_task.py    - Search query generation task
│       ├── t2_search_engine_task.py     - Search engine task
│       ├── t3_scraping_task.py          - Web scraping task
│       └── t4_procurement_report_task.py - Procurement report generation task
│
├── examples/                        - Example scripts to run individual agents or full workflows
│   ├── ex1_run_search_queries_agent.py  - Run search queries agent
│   ├── ex2_run_search_engine_agent.py   - Run search engine agent
│   └── ex3_run_procurement_report_agent.py - Run procurement report agent
|   
│
├── outputs/                         - Output directory for agent results
│   └── ai-agent-output/             - JSON outputs from agent runs
│       ├── step_1_suggested_search_queries.json - Output from search queries agent
│       ├── step_2_search_results.json           - Output from search engine agent
│       ├── step_3_scraping_results.json         - Output from web scraping agent
│       └── step_4_procurement_report.html       - Final procurement report output
│
├── tests/                           - Unit and integration tests
│   └── test.py                      - Test script
│
├── requirements.txt                 - Project dependencies
└── README.md                        - Project documentation
```

## Output Files 📁

The agents produce the following output files during execution:

```
outputs/
└── ai-agent-output/
    ├── step_1_suggested_search_queries.json - Output from search queries agent
    ├── step_2_search_results.json           - Output from search engine agent
    ├── step_3_scraping_results.json         - Output from web scraping agent
    └── step_4_procurement_report.html       - Final procurement report output (HTML format)
```

## Dependencies 📚

- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [LangChain](https://github.com/langchain-ai/langchain)
- [AgentOps](https://github.com/AgentOps-AI/agentops)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- [Requests](https://requests.readthedocs.io/)
- Python 3.10+

## Usage 🚀

### 1. Generate Search Queries 🔎

```python
from examples.ex1_run_search_queries_agent import run_search_queries_agent

results = run_search_queries_agent()
print(results)
```

### 2. Run Search Engine Agent 🌐

```python
from examples.ex2_run_search_engine_agent import run_search_engine_agent

results = run_search_engine_agent()
print(results)
```

### 3. Generate Procurement Report 📊

```python
from examples.ex3_run_procurement_report_agent import run_procurement_report_agent

results = run_procurement_report_agent()
print(results)
```

## Complete Workflow 🔄

To run the complete procurement workflow from search queries to final report:

```python
from crewai_agents.agents.a1_search_queries_agent import SearchQueriesAgent
from crewai_agents.agents.a2_search_engine_agent import SearchEngineAgent
from crewai_agents.agents.a3_scraping_agent import ScrapingAgent
from crewai_agents.agents.a4_procurement_report import ProcurementReportAgent
from crewai import Crew

# Initialize all agents
search_queries_agent = SearchQueriesAgent().create()
search_engine_agent = SearchEngineAgent().create()
scraping_agent = ScrapingAgent().create()
procurement_report_agent = ProcurementReportAgent().create()

# Create the crew with all agents
crew = Crew(
    agents=[
        search_queries_agent,
        search_engine_agent,
        scraping_agent,
        procurement_report_agent
    ],
    tasks=[
        search_queries_agent.task,
        search_engine_agent.task,
        scraping_agent.task,
        procurement_report_agent.task
    ],
    verbose=True
)

# Run the crew workflow
result = crew.kickoff(
    inputs={
        "product_category": "office chairs",
        "requirements": "ergonomic, adjustable height, under $300",
        "quantity_needed": 20
    }
)

print(result)
```

This workflow:
1. Generates optimized search queries for your product requirements
2. Searches e-commerce sites using these queries
3. Scrapes detailed product information from search results
4. Produces a comprehensive procurement report with recommendations

## License 📜

MIT License

Copyright (c) 2023-2025 Fawzy-AI-Explorer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Maintainer 👨‍💻

- [Fawzy-AI-Explorer](https://github.com/Fawzy-AI-Explorer)

## Acknowledgments 🙏

- Thanks to the [CrewAI](https://github.com/joaomdmoura/crewAI) project for providing the multi-agent framework
- Special thanks to all contributors and community members