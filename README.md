# LlamaIndex and CrewAI Agent Examples

This project provides a collection of Python scripts demonstrating how to build and orchestrate AI agents using the LlamaIndex and CrewAI frameworks. The examples range from simple, single-purpose agents to more complex, multi-agent workflows.

## Project Overview

The project is structured as a series of standalone Python scripts, each showcasing a specific feature or use case of LlamaIndex and CrewAI. The examples cover:

- **Simple Agents:** Basic agent implementation for answering queries.
- **Agents with Memory:** Adding conversational memory to agents.
- **Stateful Agents:** Persisting and restoring agent state.
- **Multi-Agent Workflows:** Orchestrating multiple agents to accomplish a complex task.

## Features

- **LlamaIndex:**
  - Simple agent creation with `FunctionAgent`.
  - Integration with Tavily for web search.
  - Managing agent memory and state.
  - Building a multi-agent research workflow.
- **CrewAI:**
  - Creating a simple multi-agent system.
  - Building a customer support analysis workflow with multiple agents.
  - Defining agents with roles, goals, and backstories.
  - Assigning tasks to agents and orchestrating their collaboration.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An `.env` file with your API keys for Google and Tavily.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/week3-llamaindex-crewai-agents.git
    cd week3-llamaindex-crewai-agents
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file in the root of the project and add your API keys:
    ```
    GOOGLE_API_KEY="your-google-api-key"
    TAVILY_API_KEY="your-tavily-api-key"
    ```

### Running the Examples

Each Python script is a standalone example that can be run from the command line:

```bash
python 1_llamaindex_simple_agent.py
python 2_llamaindex_simple_agent_memory.py
python 3_llamaindex_simple_agent_memory_restore.py
python 4_llamaindex_research_workflow_multi_agent.py
python 5_crewai_simple_multi_agent.py
python 5_crewai_customersupport_multi_agent.py
```

## File Descriptions

- **`1_llamaindex_simple_agent.py`**: A simple agent that uses LlamaIndex and Tavily to answer questions.
- **`2_llamaindex_simple_agent_memory.py`**: An extension of the simple agent that adds memory to the agent.
- **`3_llamaindex_simple_agent_memory_restore.py`**: An extension of the agent with memory that shows how to restore the agent's state from a file.
- **`4_llamaindex_research_workflow_multi_agent.py`**: A more complex example that uses multiple agents to perform a research task.
- **`5_crewai_simple_multi_agent.py`**: A simple example of a multi-agent system using CrewAI.
- **`5_crewai_customersupport_multi_agent.py`**: A more complex example of a multi-agent system using CrewAI to analyze customer support data.
- **`Homework.txt`**: A task to add more tools to the LlamaIndex agents.
- **`requirements.txt`**: The Python dependencies for the project.
- **`pyproject.toml`**: Project metadata.
- **`.env`**: For API keys.

## Homework

As an exercise, you can extend the LlamaIndex agents by adding more tools. The `Homework.txt` file contains the following task:

> Add additional tools as we did in langchain and langgraph agnets tools to this llamaindex agent modules.
