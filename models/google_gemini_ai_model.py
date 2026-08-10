import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

class GoogleGeminiAI(DeepEvalBaseLLM):
    """Class to Implement Gemini AI Studio for deepeval"""
    def __init__(self, model: ChatGoogleGenerativeAI):
        self.model: ChatGoogleGenerativeAI = model
        super().__init__()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.model
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.model
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Gemini API model"

# Factory function to create model instances
def get_gemini_model(model_name: str = "gemini-2.5-flash", temperature: float = 0):
    """Factory function to create GoogleGeminiAI instances"""
    custom_model_gemini = ChatGoogleGenerativeAI(
        model = model_name,
        google_api_key = os.getenv("GEMINI_API_KEY"),
        temperature = temperature,
    )
    return GoogleGeminiAI(model = custom_model_gemini)


#
# gemini_model = GoogleGeminiAI(model = custom_model_gemini)
#
# print(gemini_model.generate("Write me a joke"))
