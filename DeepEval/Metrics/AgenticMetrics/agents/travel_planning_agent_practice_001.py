# ==============================================================================
# AGENT 3 — Travel Planning Agent
# Metrics: PlanAdherence + PlanQuality (both trace-only)
#
# Why good for these task_completion?
#   • Travel planning is multi-step: research → hotels → restaurants → itinerary.
#   • A good system prompt with explicit reasoning/thinking exposes the "plan"
#     that DeepEval extracts from the trace.
#   • PlanQuality: judges if the plan was well-structured for the task.
#   • PlanAdherence: judges if execution matched what was planned.
#   • Giving it a contradictory or rushed query will produce deviations.
# ==============================================================================

from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def search_flights(origin:str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date"""
    return f"Flights from {origin} to {destination} on {date}: [Air India 09:00 AM $250, Indigo 02:00 PM $180]"

@tool
def search_hotels(city: str, check_in: str, budget_per_night: float = None) -> str:
    """Search for hotels in a city. Optionally filter by budget per night."""
    result = f"Hotels in {city} for check-in {check_in}"
    if budget_per_night:
        result += f" under ${budget_per_night}/night"
    return result + ": [The Grand Hotel $120/night, City Inn $75/night]"\

@tool
def get_tourist_attractions(city: str, category: str = "all") -> str:
    """Get top tourist attractions in a city. Category - museums, food, outdoor, all."""
    attractions = {
        "Paris": "Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Montmartre",
        "Tokyo": "Senso-ji Temple, Shibuya Crossing, TeamLab, Mount Fuji Day trip",
        "London": "Big Ben, British Museum, Tower of London, Hyde Park",
    }
    return attractions.get(city, f"Top attractions in {city}: City Centre, Local Museum, Main Market")

@tool
def create_day_itinerary(city: str, day_number: int, activities: str) -> str:
    """Create a structured day-by-day itinerary entry for a city."""
    return f"Day {day_number} in {city}: {activities}"

# System prompt with explicit planning instructions - helps PlanQuality/PlanAdherence
TRAVEL_SYSTEM_PROMPT = """
You are a structured travel planning assistant.

When given a travel request, ALWAYS follow this explicit Plan:
Step 1: Search for flights to the destination.
Step 2: Search for hotels within the user's budget.
Step 3: Get tourist attractions for the city.
Step 4: Create a day-by-day itinerary combining hotels and attractions.
Step 5: Present a complete travel summary.

Think step-by-step and state your plan before executing it.
"""

travel_agent = create_agent(
    model = "gpt-4o",
    tools = [search_flights, search_hotels, get_tourist_attractions, create_day_itinerary],
    system_prompt = TRAVEL_SYSTEM_PROMPT
)