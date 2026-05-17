from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams
from deepeval.metrics import ConversationalGEval

load_dotenv()

def test_professionalism():
    professionalism_metric = ConversationalGEval(
        name = "Professionalism",
        criteria = "Determine whether the assistant answered the questions of the user in a professional and polite manner."
    )

    conversation_example1 = ConversationalTestCase(
        turns = [
            Turn(role = 'user', content = "Is Python an interpreted language?"),
            Turn(role = 'assistant', content = "Of course! How could you not know?"),
            Turn(role = 'user', content = "What about C++?"),
            Turn(role = 'assistant', content = "Damn... You really know nothing. C++ is compiled man.")
        ]
    )

    conversation_example2 = ConversationalTestCase(
        turns = [
            Turn(role = 'user', content = "Is Python an interpreted language?"),
            Turn(role = 'assistant', content = "Yes, Python is an interpreted language."),
            Turn(role = 'user', content = "What about C++?"),
            Turn(role = 'assistant', content = "C++ is a compiled language")
        ]
    )

    evaluate(test_cases=[conversation_example1, conversation_example2], metrics=[professionalism_metric])