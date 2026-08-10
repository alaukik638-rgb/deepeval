# ==============================================================================
# AGENT 1 — Math Calculator Agent
# metrics: TaskCompletion + StepEfficiency (both trace-only)
#
# Why good for these task_completion?
#   • Simple arithmetic tools let deepeval clearly judge whether the task
#     was completed (TaskCompletion) and whether extra/redundant steps were
#     taken (StepEfficiency).
#   • Ask it to "add then multiply" — it should call tools in sequence.
#   • Ask it to divide by zero — it will fail, giving a low TaskCompletion score.
# ==============================================================================

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from typing import Any
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4.1",
    temperature = 0,
)

@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

@tool
def divide(a: float, b: float) -> float | str:
    """Divide a by b. Return error string if b is zero."""
    if b == 0:
        return "Error: Division by 0 is undefined"
    return a / b

# agent_input: dict[str, Any] = {
#     "messages": [
#             HumanMessage(content="What is (15+5) * 3?")
#     ]
# }
# response = math_agent.invoke(agent_input)
#
# print(response["messages"][-1].content)

TOOL_DESCRIPTIONS = {
    "add": add.description,
    "multiply": multiply.description,
    "divide": divide.description,
}

math_agent = create_agent(model = llm, tools = [add, multiply, divide])