# STEP 0 env and import libraries

from dotenv import load_dotenv
load_dotenv()
import os
load_dotenv()
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

from crewai import LLM
import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from google.genai import types

from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name="CrewAI-agent-proeject", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)
# STEP 1: LLM


llm = LLM(
    model="gemini-2.5-flash-lite",temperature=0.0,
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        )  # Disables thinking
    ),
)
# STEP 2:  Agent definion
research_agent = Agent(
    role="Research Specialist",
    goal="Research interesting facts about the topic: {topic}",
    backstory="You are an expert at finding relevant and factual data.",
    tools=[SerperDevTool()],
    verbose=True,
    llm=llm
)

writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research",
    backstory="You are skilled at writing engaging summaries based on provided content.",
    llm=llm,
    verbose=True,
)

# STEP 3:  Assign tasks to agnets
task1 = Task(
    description="Find 3-5 interesting and recent facts about {topic} as of year 2025.",
    expected_output="A bullet list of 3-5 facts",
    agent=research_agent,
     output_file="./task1_output.txt"
)

task2 = Task(
    description="Write a 100-word blog post summary about {topic} using the facts from the research.",
    expected_output="A blog post summary",
    agent=writer_agent,
    context=[task1],
    output_file="./final_output.txt"
)

# STEP 4:  Create the crew orchestrator
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[task1, task2],
    verbose=True,
    memory=True,
    embedder={
        "provider": "google",
        "config": {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "model": "text-embedding-001"
        }
    }
)


# STEP 5:  Run the crew

# crew.kickoff(inputs={"topic": "The future of electrical vehicles"})
crew.kickoff(inputs={"topic": "What is the revenue outlook in IT sector?"})
