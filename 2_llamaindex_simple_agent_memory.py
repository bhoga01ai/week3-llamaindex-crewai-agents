# This is the a simple agent based on llamaindex framework
# Agents 
# Agents are the core building blocks of the LlamaIndex framework. They are responsible for interacting with the user, receiving input, and producing output. The agents are also responsible for coordinating the actions of the various components of the framework.

# Agent Types
# There are two main types of agents in LlamaIndex:
# - ReAct Agent: This agent is responsible for reasoning about the user's input and then taking action. It is the most common agent type and is used in the example code provided.
# - Tree-of-Thought Agent: This agent is responsible for reasoning about the user's input and then taking action. It is a more complex agent that is used in the example code provided.

# Agent Initialization
# The agent is initialized with a set of tools. The tools are the components of the framework that the agent can use to perform actions. The tools are responsible for interacting with the outside world and returning the results of the actions.

# Agent Execution

# STEPS
# 0  API keys  and import libraries
# 1. LLM
# 2. Tools - one search tool
# 3. Agentframework - FunctionAgent from LlamaIndex  - Orchestrator
# 4. ADD state to agent.  - Context(agent)
# 5. Agent Execution
# 6 .save the STATE


# STEP 0 - env and import libraries

from turtle import reset
from dotenv import load_dotenv
load_dotenv()

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types
from tavily import AsyncTavilyClient
from llama_index.core.agent.workflow import FunctionAgent
import asyncio
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
#STEP 1. LLM  - openAI

# llm = OpenAI(model="gpt-4o-mini", temperature=0.5)

llm = GoogleGenAI(
    model="gemini-2.5-flash",temperature=0.5,
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        )  # Disables thinking
    ),
)
# LLM test
# prompt="who are you"
# response=llm.complete(prompt)
# print(response)

# 2. Tools - one search tool
# Tavily API
async def search_web(query: str) -> str:
    """
    Search the web using Tavily API and return results as a string.
    Args:
        query (str): The search query to execute
    Returns:
        str: Search results from Tavily API
    Raises:
        TavilyError: If the API request fails
    """
    try:
        client = AsyncTavilyClient()
        result = await client.search(query)
        return str(result)
    except Exception as e:
        # Handle any errors that occur during the search
        error_message = f"Error occurred during web search: {str(e)}"
        print(f"Search error: {error_message}")
        return f"Search failed: {error_message}"

# STEP 3
# create an functon agent 
agent = FunctionAgent(
    llm=llm,
    tools=[search_web],
    system_prompt="""You are a helpful assistant with access to web search capabilities. 
    You can search the web for current information including:
    - Weather forecasts and current conditions
    - Latest news and events
    - Real-time data and updates
    - General information and facts
    
    When a user asks for information, especially current/live data like weather, 
    you should use your web search tool to find the most up-to-date information.
    Always try to search for the information before saying you cannot provide it.""",
)
# STEP 4 ADD state to agent.

ctx = Context(agent)
# STEP 5
# Execute the agent with a query

async def main():
    """Main async function to run the agent."""
    while True:
        user_msg = input("user: ")
        if user_msg.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        try:
            response = await agent.run(user_msg=user_msg,ctx=ctx)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
    
    # Save the context state
    try:
        ctx_dict = ctx.to_dict(serializer=JsonSerializer())
        
        # Save to file manually
        import json
        with open("agent_state.json", "w") as f:
            json.dump(ctx_dict, f, indent=2)
        
        print("Agent state saved to agent_state.json")
    except Exception as e:
        print(f"Error saving state: {e}")






