import os
from dotenv import load_dotenv

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

instruction_message = """
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
"""


if __name__ == '__main__':
    # create a prompt template
    instructions = SystemMessage(content=instruction_message)
    question = HumanMessage(content="What is the weather like?")

    # initiate the Ollama client
    chat_llm = ChatOllama(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model="gemma:7b",
        temperature=0.0
    )

    # execute the LLM
    response = chat_llm.invoke([instructions, question])
    print(response.content)
