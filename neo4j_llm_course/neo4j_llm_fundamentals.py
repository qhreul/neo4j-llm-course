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

Context: {context}
Question: {question}
"""


if __name__ == '__main__':
    # create a prompt template
    prompt = PromptTemplate(
        template=instruction_message,
        input_variables=["context", "question"]
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

    # providing some context to the LLM - minimal / manual RAG
    current_weather = """
        {
            "surf": [
                {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
                {"beach": "Polzeath", "conditions": "Flat and calm"},
                {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
            ]
        }"""

    # execute the LLM
    response = chat_chain.invoke({
        "context": current_weather,
        "question": "What is the weather like today in Watergate Bay?"
    })
    print(response)
