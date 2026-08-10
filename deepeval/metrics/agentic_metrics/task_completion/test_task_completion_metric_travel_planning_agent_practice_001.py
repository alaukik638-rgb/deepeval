from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics.agentic_metrics.agents.travel_planning_agent_practice_001 import travel_agent
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()

def test_task_completion_metric_travel_planning_agent():
    @observe()
    def run_travel_planning_agent(user_input: str) -> str:
        result = travel_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            }
        )
        return result["messages"][-1].content

    dataset = EvaluationDataset(
        goldens = [
            Golden(
                input = "Plan a 3-day trip to Paris from Delhi on 15th July with a budget of $150/night for hotels.",
                multimodal = False,
            ),
            Golden(
                input = "Just find me hotels in Tokyo for next weekend.",
                multimodal = False,
            ),
            Golden(
                input = "Plan a 7-day multi-city Europe trip covering London and Paris departing from Mumbai on August 1st, budget $200/night.",
                multimodal = False,
            )
        ]
    )

    task_completion_metric = TaskCompletionMetric(
        threshold = 0.7,
        model = "gpt-4o",
        include_reason = True,
        task = None,
        verbose_mode = False,
        async_mode = True,
        strict_mode = False,
    )

    test_cases = []
    for golden in dataset.goldens:
        output = run_travel_planning_agent(golden.input)
        test_cases.append(
            LLMTestCase(
                input = golden.input,
                actual_output = output,
            )
        )

    evaluate(test_cases, metrics = [task_completion_metric])

"""
metrics Summary

  - ✅ Task Completion (score: 0.7, threshold: 0.7, strict: False, evaluation model: gpt-4o, 
  reason: The system is ready to search for hotels in Tokyo for the upcoming weekend, indicating partial completion of the task. 
  However, it has not yet provided specific hotel options or confirmed availability, 
  which is necessary to fully achieve the desired outcome., error: None)

For test case:

  - input: Just find me hotels in Tokyo for next weekend.
  - actual output: To find hotels in Tokyo for next weekend, I will follow these steps:

1. Search for hotels in Tokyo with your specified budget (if any).
   
Please let me know if you have a specific budget per night, or I will proceed without any budget constraints.
  - expected output: None
  - context: None
  - retrieval context: None

======================================================================

metrics Summary

  - ✅ Task Completion (score: 0.95, threshold: 0.7, strict: False, evaluation model: gpt-4o, 
  reason: The system successfully planned a 3-day trip to Paris from Delhi, including flight details, 
  hotel accommodations within the specified budget, and a comprehensive list of activities and sightseeing options. 
  The only minor shortcoming was the lack of specific details on transportation within Paris., error: None)

For test case:

  - input: Plan a 3-day trip to Paris from Delhi on 15th July with a budget of $150/night for hotels.
  - actual output: Here is your complete 3-day travel itinerary for Paris:

### Day 1: Arrival and Sightseeing
- **Flight**: Arrive in Paris with Indigo at 02:00 PM.
- **Hotel**: Check in at **City Inn** ($75/night).
- **Activities**:
  - Visit the **Eiffel Tower**.
  - Explore the **Louvre Museum**.

### Day 2: Cultural Exploration
- **Activities**:
  - Discover the **Notre-Dame Cathedral**.
  - Wander through **Montmartre**, a historic and artistic district.

### Day 3: Leisure and Local Experience
- **Activities**:
  - Enjoy a leisure day and explore local cafes, savor French cuisine, and soak in the Parisian atmosphere.

This itinerary takes into account your budget and includes your flight details, accommodations, 
and a mix of iconic attractions and leisurely activities.

Enjoy your trip to Paris! If you need any more assistance or modifications, feel free to ask.
  - expected output: None
  - context: None
  - retrieval context: None

======================================================================

metrics Summary

  - ✅ Task Completion (score: 0.95, threshold: 0.7, strict: False, evaluation model: gpt-4o, 
  reason: The system successfully planned a 7-day multi-city trip covering both London and Paris, 
  adhering to the budget for accommodation and including a flight between the cities. 
  The itinerary also featured activities and sightseeing options, closely aligning with the desired task. 
  However, the departure flight from Mumbai was not explicitly mentioned, 
  which slightly affects the completeness of the plan., error: None)

For test case:

  - input: Plan a 7-day multi-city Europe trip covering London and Paris departing from Mumbai on August 1st, budget $200/night.
  - actual output: ### 7-Day Europe Trip Itinerary: London and Paris

**Day 1: London**
- Arrive in London
- Check-in at The Grand Hotel ($120/night)
- Visit Hyde Park
- Evening walk around Big Ben

**Day 2: London**
- Visit British Museum
- Explore Tower of London
- Free time for shopping

**Day 3: London to Paris**
- Visit more sights or enjoy leisure activities of choice
- Departure to Paris

**Flight from London to Paris on August 4th:**
- Air India: Departing at 09:00 AM, Price: $250
- Indigo: Departing at 02:00 PM, Price: $180

**Day 4: Paris**
- Arrive in Paris
- Check-in at The Grand Hotel ($120/night)
- Visit Eiffel Tower
- Evening at a Parisian café

**Day 5: Paris**
- Explore Louvre Museum
- Visit Notre-Dame Cathedral

**Day 6: Paris**
- Explore Montmartre
- Leisure activities

**Day 7: Paris**
- Free time
- Departure

### Travel Summary
Your trip begins with a flight from Mumbai to London on August 1st, where you will enjoy three days exploring iconic attractions. 
You will then travel to Paris on August 4th to continue your adventure until your departure on the 7th day. 
Accommodations in both cities have been arranged to fit within your budget, 
enhancing your travel experience with comfort and convenience.

Feel free to ask for additional details or modifications!
  - expected output: None
  - context: None
  - retrieval context: None

"""