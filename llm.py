import os
from dotenv import load_dotenv
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

PROVIDER_LIST=["aws","mistral","ollama"]
AWS_MODEL_LIST=["anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-5-sonnet-20241022-v2:0"]

load_dotenv()
PROVIDER = os.getenv("USE_PROVIDER", "aws")
MODEL_NAME = os.getenv("USE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")

def llm_from(provider=PROVIDER,model_name=MODEL_NAME):
    """Initialize and return a language model based on the specified provider and model name.
    providers should be one of "aws", "mistral" or "ollama"."""
    if provider not in PROVIDER_LIST:
        raise ValueError("choose from \"aws\", \"mistral\" or \"ollama\"")
    if provider == "aws":

        return ChatBedrock(model_id=model_name, region="us-west-2", temperature=0)
    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        return ChatMistralAI(
            model=model_name,
            api_key=api_key,
            temperature=0
        )
    if provider == "ollama":
        return ChatOllama(model=model_name, temperature=0)
