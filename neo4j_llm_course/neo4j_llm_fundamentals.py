import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

load_dotenv()

prompt = """
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
"""


if __name__ == '__main__':
    # create a prompt template
    prompt = PromptTemplate(
        template=prompt, input_variables=["fruit"]
    )

    # initiate the Ollama client
    llm = Ollama(base_url=os.getenv('OLLAMA_ENDPOINT'), model="phi:2.7b")

    # execute the LLM
    response = llm.invoke(prompt.format(fruit="apple"))
    print(response)
