import os
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

instruction_message = """
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Question: {question}
"""


if __name__ == '__main__':
    # create a prompt template
    prompt = PromptTemplate(
        template=instruction_message,
        input_variables=["question"]
    )

    # initiate the Ollama client
    chat_llm = ChatOllama(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model="gemma:7b",
        temperature=0.0
    )

    # initiate the LLMChain
    chat_chain = LLMChain(
        llm=chat_llm,
        prompt=prompt
    )

    # execute the LLM
    response = chat_chain.invoke({"question": "What is the weather like today?"})
    print(response)
