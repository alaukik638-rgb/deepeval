from deepeval.models import DeepEvalBaseLLM
import requests

class OllamaLLM(DeepEvalBaseLLM):
    def __init__(self, model="llama3:8b"):
        self.model = model
        print(f"Using model: {self.model}")

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout = 60
        )

        if response.status_code != 200:
            raise Exception(f"Ollama error {response.text}")
        return response.json().get("response", "")

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model