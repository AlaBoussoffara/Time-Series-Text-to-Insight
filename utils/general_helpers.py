import os
from typing import Optional

from dotenv import load_dotenv
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.messages import BaseMessage
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from utils.api_call_counter import wrap_llm_with_api_call_counter
from utils.messages import AgentMessage

PROVIDER_LIST = ["aws", "mistral", "ollama"]
AWS_MODEL_LIST = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
]

load_dotenv()
PROVIDER = os.getenv("USE_PROVIDER", "aws")
MODEL_NAME = os.getenv("USE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")


def llm_from(
    provider: str = PROVIDER,
    model_name: str = MODEL_NAME,
    *,
    agent_name: str | None = None,
):
    """Initialize and return a language model based on the specified provider and model name."""
    if provider not in PROVIDER_LIST:
        raise ValueError('choose from "aws", "mistral" or "ollama"')
    if provider == "aws":
        llm = ChatBedrock(model_id=model_name, region_name="us-west-2")
    elif provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        llm = ChatMistralAI(
            model=model_name,
            api_key=api_key,
            temperature=0
        )
    elif provider == "ollama":
        llm = ChatOllama(model=model_name, temperature=0)
    else:
        raise ValueError('choose from "aws", "mistral" or "ollama"')
    return wrap_llm_with_api_call_counter(llm, agent_name=agent_name)


def format_message(message: BaseMessage) -> tuple[str, Optional[str], str]:
    """Return (name, output_type, output_content) for agent/tool/system messages."""
    name = message.name or getattr(message, "type", message.__class__.__name__)
    if isinstance(message, AgentMessage):
        structured = message.structured_output
        output_type = structured.get("output_type")
        output_content = structured.get("output_content", message.content)
        return name, output_type, str(output_content)
    structured = getattr(message, "structured_output", None)
    if isinstance(structured, dict):
        output_type = structured.get("output_type")
        output_content = structured.get("output_content")
    else:
        output_type = getattr(message, "output_type", None)
        output_content = getattr(message, "output_content", getattr(message, "content", ""))
    return name, output_type, str(output_content)


def stream_graph(
    compiled_graph,
    state,
    *,
    log: bool = True,
    recursion_limit: Optional[int] = 50,
) -> list[BaseMessage]:
    """Aggregate global message history as the graph executes, logging updates when requested."""
    history: list[BaseMessage] = list(state.get("global_messages_history", []))
    seen_ids = {id(message) for message in history}
    config = {"recursion_limit": recursion_limit} if recursion_limit else None
    for event in compiled_graph.stream(state, config=config):
        for node, payload in event.items():
            messages = payload.get("global_messages_history")
            if not messages:
                continue
            for message in messages:
                message_id = id(message)
                if message_id in seen_ids:
                    continue
                history.append(message)
                seen_ids.add(message_id)
            if node == "__end__":
                return history
            last_message = messages[-1]
            if log:
                name, output_type, output_content = format_message(last_message)
                if output_type:
                    print(f"[{name}] ({output_type}): {output_content}")
                else:
                    print(f"[{name}]: {output_content}")
    return history
