# ==============================================================================
# AGENT 2 — E-Commerce Customer Support Agent
# Metrics: ArgumentCorrectness + ToolCorrectness (both use LLMTestCase)
#
# Why good for these task_completion?
#   • Multiple tools with distinct argument signatures expose argument mistakes.
#   • We can define expected_tools to check tool selection correctness.
#   • Deliberately ambiguous queries will cause wrong tool selection or bad args,
#     producing lower scores and showing metric range.
# ==============================================================================

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def check_order_status(order_id: str) -> str:
    """Check the current status of an order given an order id"""
    fake_db ={
        "ORD-001": "Shipped - Expected delivery - Tomorrow",
        "ORD-002": "Processing - Will ship in 2 days",
        "ORD-999": "Order not found"
    }
    return fake_db.get(order_id, "Order not found")

@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund for an order. Requires order_id and reason"""
    return f"Refund initiated for an order {order_id}. Reason: {reason}. You will receive confirmation in 24 hours"

@tool
def search_products(query: str, max_price: float = None) -> str:
    """Search for products in catalog. Optionally filter by max_price."""
    results = f"found products matching '{query}'"
    if max_price:
        results += f" under ${max_price}"
    return results + ": [Blue Running Shoes $49, Red Sneakers $39]"

support_agent = create_agent(
    model = "gpt-4o-mini",
    tools = [check_order_status, process_refund, search_products]
)