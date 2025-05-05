# CrewAI Procurement Agents 🤖

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.1.30+-orange.svg)](https://github.com/joaomdmoura/crewAI)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.335+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Fawzy-AI-Explorer/NLP-Tea/issues)
[![Stars](https://img.shields.io/github/stars/Fawzy-AI-Explorer/NLP-Tea?style=social)](https://github.com/Fawzy-AI-Explorer/NLP-Tea/stargazers)

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
   ```bash
   git clone https://github.com/Fawzy-AI-Explorer/NLP-Tea.git
   cd NLP-Tea/04-crewai-agents/4.1-AI\ Agents\ using\ CrewAI\ \(\ Abu\ Bakr\ Soliman\)/crewai-agents
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # For Windows
   python -m venv venv
   source venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
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

workflow:
1. Generates optimized search queries for your product requirements
2. Searches e-commerce sites using these queries
3. Scrapes detailed product information from search results
4. Produces a comprehensive procurement report with recommendations

## License 📜

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments 🙏

- Thanks to the [Abu Bakr Soliman](https://www.linkedin.com/in/bakrianoo/) for this [crash course](https://www.youtube.com/watch?v=DDR4A8-MLQs&t=1s)