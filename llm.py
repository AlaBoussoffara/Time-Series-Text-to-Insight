from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()

def llm_from(provider=None):
    "choose provider from mistral-cloud (using public API) key or local model (ollama)"
    if provider == "mistral-cloud":
        cloud_model = os.getenv("MISTRAL_CLOUD_MODEL")
        api_key = os.getenv("MISTRAL_API_KEY")
        return ChatMistralAI(
            model=cloud_model,
            api_key=api_key,
            temperature=0
        )
    
    if provider == "ollama":
        local_model = os.getenv("LOCAL_MODEL")
        return ChatOllama(model=local_model, temperature=0)
    
    raise ValueError("choose from \"mistral-cloud\" or \"ollama\"")