import os
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama

load_dotenv()

instruction_message = """
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Chat History: {chat_history}
Context: {context}
Question: {question}
"""


if __name__ == '__main__':
    # create a prompt template
    prompt = PromptTemplate(
        template=instruction_message,
        input_variables=["chat_history", "context", "question"]
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
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
        prompt=prompt,
        memory=memory
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
    print(response["text"])

    response = chat_chain.invoke({
        "context": current_weather,
        "question": "Where I am?"
    })
    print(response["text"])
