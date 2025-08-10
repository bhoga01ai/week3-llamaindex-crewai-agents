# This is a academic research workflow agent based on llamaindex framework
#The Research Agent will use Python functions as tools.
# 1. search_web uses Gemini with Google Search to search the web for information on the given topic.
# 2. record_notes saves research found on the web to the state so that the other tools can use it.
# 3. write_report writes the report using the information found by the ResearchAgent
# 4. review_report reviews the report and provides feedback.

# STEPS
# 0  API keys  and import libraries
# 1. LLM
# 2. Tools - google search tool and research tools (search_web, record_notes, write_report, review_report)
# 3. Build individual agents 
# 4. Build a multi agent orchestrator
# 5. Agent Execution
# 6 .save the report

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
import json
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.core.agent.workflow import AgentWorkflow

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

# STEP 2. Tools - google search tool

google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)

llm_with_search = GoogleGenAI(
    model="gemini-2.5-pro",
    generation_config=types.GenerateContentConfig(tools=[google_search_tool])
)

# A simple test
# response = llm_with_search.complete("What's the weather like today in New Delhi India?")
# print(response)

async def search_web(ctx: Context, query: str) -> str:
    """Useful for searching the web about a specific query or topic"""
    response = await llm_with_search.acomplete(f"""Please research given this query or topic,
    and return the result\n<query_or_topic>{query}</query_or_topic>""")
    return response

async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.store.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.store.set("state", current_state)
    return "Notes recorded."

async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.store.get("state")
    current_state["report_content"] = report_content
    await ctx.store.set("state", current_state)
    return "Report written."

async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.store.get("state")
    current_state["review"] = review
    await ctx.store.set("state", current_state)
    return "Report reviewed."

# Step 3 Build individual agents.

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic."
    ),
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report on a given topic.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent."
    ),
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review a report and provide feedback. "
        "Your feedback should either approve the current report or request changes for the WriteAgent to implement."
    ),
    llm=llm,
    tools=[review_report],
    can_handoff_to=["ResearchAgent","WriteAgent"],
)

# STEP 4. Build a multi agent orchestrator

agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

# STEP 5 Agent execution

research_topic = """Write me a report on the history of the web.
Briefly describe the history of the world wide web, including
the development of the internet and the development of the web,
including 21st century developments"""

# Remove this line from global scope:
# handler = agent_workflow.run(user_msg=research_topic)

async def main():
    # Move the workflow execution here:
    handler = agent_workflow.run(user_msg=research_topic)
    
    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if (isinstance(event, AgentInput
            and hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent)
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
    
    # Move these lines inside the main() function:
    print("--------final report and review --------")
    state = await handler.ctx.store.get("state")
    print("Report Content:\n", state["report_content"])
    print("\n------------\nFinal Review:\n", state["review"])

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())