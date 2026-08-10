import pytest
from langchain_core.messages import AIMessage
from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.metrics.argument_correctness.template import ArgumentCorrectnessTemplate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics.agentic_metrics.agents.math_calculator_agent_practice_001 import math_agent, TOOL_DESCRIPTIONS
from dotenv import load_dotenv

load_dotenv()

def extract_tool_calls(response: dict) -> list[ToolCall]:
    deepeval_tool_calls = []
    for message in response["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tc in message.tool_calls:
                # print(f"  RAW TC KEYS : {tc.keys()}")
                # print(f"  RAW TC FULL : {tc}")
                deepeval_tool_calls.append(
                    ToolCall(
                        name = tc["name"],
                        description = TOOL_DESCRIPTIONS.get(
                            tc["name"], "No description"
                        ),
                        input_parameters = tc["args"],
                    )
                )
    return deepeval_tool_calls

# @pytest.mark.parametrize(
#     "user_input", [
#     "What is (15 + 5) * 3?",
#     "What is 10 divided by 0?",
#     "What is 100 divided by 4?",
#     ]
# )
# def test_argument_correctness(user_input):
#     response = math_agent.invoke(
#         {
#             "messages": [{
#                 "role": "user",
#                 "content": user_input,
#             }]
#         }
#     )
#
#     # tool_calls = extract_tool_calls(response)
#
#     # print(f"\n{'=' * 60}")
#     # print(f"INPUT        : {user_input}")
#     # print(f"ACTUAL OUTPUT: {response['messages'][-1].content}")
#     # print(f"TOOL CALLS   : {len(tool_calls)}")
#     # for t in tool_calls:
#     #     print(f"  → name       : {t.name}")
#     #     print(f"  → description: {t.description}")
#     #     print(f"  → input      : {t.input_parameters}")
#     # print(f"{'=' * 60}\n")
#
#     test_case = LLMTestCase(
#         input = user_input,
#         actual_output = response["messages"][-1].content,
#         tools_called = extract_tool_calls(response),
#     )
#
#     metric = ArgumentCorrectnessMetric(
#         threshold = 0.7,
#         model = "gpt-4.1",
#         include_reason = True,
#         verbose_mode = False,
#         strict_mode = False,
#         async_mode = True,
#         evaluation_template = ArgumentCorrectnessTemplate,
#     )
#
#     evaluate(test_cases = [test_case], metrics = [metric])

def test_argument_correctness():
    user_inputs = [
        "What is (15 + 5) * 3?",
        "What is 10 divided by 0?",
        "What is 100 divided by 4?",
    ]
    responses = []
    for user_input in user_inputs:
        responses.append(math_agent.invoke(
            {
                "messages": [{
                    "role": "user",
                    "content": user_input,
                }]
            }
        ))

    test_cases = []
    for user_input, response in zip(user_inputs, responses):
        test_cases.append( LLMTestCase(
            input = user_input,
            actual_output = response["messages"][-1].content,
            tools_called = extract_tool_calls(response),
        ))

    metric = ArgumentCorrectnessMetric(
        threshold = 0.7,
        model = "gpt-4.1",
        include_reason = True,
        verbose_mode = False,
        strict_mode = False,
        async_mode = True,
        evaluation_template = ArgumentCorrectnessTemplate,
    )

    evaluate(test_cases = test_cases, metrics = [metric])