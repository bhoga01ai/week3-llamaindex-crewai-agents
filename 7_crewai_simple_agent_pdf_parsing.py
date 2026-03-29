 STEP 0 env and import libraries

from dotenv import load_dotenv
load_dotenv()
import os
load_dotenv()
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

from crewai import LLM
import os
from crewai import Agent, Task, Crew
from google.genai import types

from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name="CrewAI-invoice-parser-agent",  # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)
# STEP 1: LLM


llm = LLM(
    model="gemini-2.5-flash",temperature=0.0,
    generation_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        )  # Disables thinking
    ),
)
## Tool
from crewai_tools import PDFSearchTool

# Initialize the tool with a specific PDF path for exclusive search within that document
pdf_invoice_tool = PDFSearchTool(pdf='/Users/bhogaai/week03-saturday-llamaindex-crewai/week3-llamaindex-crewai-agents/sample_invoice.pdf')


# STEP 2:  Agent definion
invoice_parser_agent = Agent(
    role="Invoice Parser Agent",
    goal="Use pdf_invoice_tool to search the invoice PDF and answer the user's question accurately.",
    backstory="You are an expert at parsing invoices to extract relevant information. Use the pdf_invoice_tool to find relevant details and then answer the question.",
    tools=[pdf_invoice_tool],
    verbose=True,
    llm=llm
)

# STEP 3:  Assign tasks to agnets
task1 = Task(
    description=(
        "Answer the user's question about the invoice PDF. "
        "Question: {question}. "
        "Use the pdf_invoice_tool to locate relevant information and provide a concise, direct answer. "
        "If the answer is not present in the PDF, respond with 'Not found in the provided PDF.'"
    ),
    expected_output="Direct answer grounded in the PDF content, including field names and values when applicable.",
    agent=invoice_parser_agent,
)


# STEP STEP 4:  Create the crew orchestrator
crew = Crew(
    agents=[invoice_parser_agent],
    tasks=[task1],
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
while True:
    user_input = input("Enter your question: ")
    if user_input.lower() == "exit":
        break
    response = crew.kickoff(inputs={"question": user_input})
    print(response)