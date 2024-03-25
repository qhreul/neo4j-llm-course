import os
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.chat_models.ollama import ChatOllama

load_dotenv()

instruction_message = """
You are a movie expert. You find movies from a genre or plot.

Chat History:{chat_history}
Question:{input}
"""


if __name__ == '__main__':
    # create a prompt template
    prompt = PromptTemplate(
        template=instruction_message,
        input_variables=["chat_history", "input"]
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
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

    # initiate the tools
    tools = [
        Tool.from_function(
            name="Movie Expert",
            description="For when you need to chat about movies. The question will be a string. Return a string.",
            func=chat_chain.run,
            return_direct=True
        )
    ]

    # "ReAct" agent stands for "reasoning and acting"
    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(chat_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        max_iterations=3,
        verbose=True,
        handle_parsing_errors=True)

    while True:
        question = input("> ")
        response = agent_executor.invoke({"input": question})
        print(response["output"])
