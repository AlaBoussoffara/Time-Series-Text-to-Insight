from pathlib import Path

import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from llm import llm_from

PERSIST_DIR = Path(".chainlit_memory")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

USERS = {"demo": "demo123"}


def build_memory(session_id: str) -> ConversationBufferMemory:
    connection = f"sqlite:///{PERSIST_DIR / 'chat_history.db'}"
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection=connection,
    )
    return ConversationBufferMemory(chat_memory=history, return_messages=True)


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if USERS.get(username) != password:
        return None
    return cl.User(identifier=username)


@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    session_id = user.identifier if user else "anonymous"
    cl.user_session.set("memory", build_memory(session_id))


llm = llm_from("mistral-ollama")

@cl.on_message
async def main(message: cl.Message):
    user = cl.user_session.get("user")
    session_id = user.identifier if user else "anonymous"
    memory: ConversationBufferMemory = cl.user_session.get("memory") or build_memory(session_id)
    cl.user_session.set("memory", memory)

    history = memory.load_memory_variables({}).get("history", [])
    conversation = history + [HumanMessage(content=message.content)]

    msg = await cl.Message(content="").send()
    response_chunks: list[str] = []

    async for chunk in llm.astream(conversation):
        token = chunk.content or ""
        if token:
            response_chunks.append(token)
            await msg.stream_token(token)

    await msg.update()

    memory.save_context({"input": message.content}, {"output": "".join(response_chunks)})
