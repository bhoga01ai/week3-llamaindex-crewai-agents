# STEP 0 env and import libraries

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

from crewai import LLM
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
pdf_contract_tool = PDFSearchTool(pdf='/Users/bhogaai/week03-saturday-llamaindex-crewai/week3-llamaindex-crewai-agents/purchase_terms_conditions.pdf')

# STEP 2:  Agent definion
invoice_parser_agent = Agent(
    role="Invoice-Contract Reconciliation Agent",
    goal="Identify and report discrepancies between the invoice and the contract PDFs using the provided tools.",
    backstory="You validate invoices against contract terms by extracting and comparing relevant fields from both documents.",
    tools=[pdf_invoice_tool, pdf_contract_tool],
    verbose=True,
    llm=llm
)

# STEP 3:  Assign tasks to agnets
task1 = Task(
    description=(
        "Analyze both the invoice PDF and the contract PDF using the available tools to extract comparable data, "
        "then produce a structured discrepancy report. Compare at least: "
        "supplier/vendor name, invoice number, invoice date, total amount, taxes, currency, payment terms, due date, "
        "late fees, discounts, purchase order or reference numbers, and line items (description, quantity, unit price, subtotal). "
        "For each field, state whether it matches; for mismatches include the values from both documents and a short rationale. "
        "For line items, align items by description and report differences in quantity, unit price, or totals, as well as missing or extra items."
        "output only the discrepancies for the line items."
    ),
    expected_output=(
        "A JSON object with keys: matched_fields (list of field names), "
        "discrepancies (list of objects with keys: field, invoice_value, contract_value, severity, rationale), "
        "missing_in_invoice (list), missing_in_contract (list), line_item_discrepancies (list of objects with keys: "
        "description, issue, invoice_value, contract_value, rationale)."
    ),
    output_file="invoice_contract_reconciliation.json",
    output_format="json",
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
response = crew.kickoff()
print(response)
