import os
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama

load_dotenv()

if __name__ == '__main__':
    # initiate the Ollama client
    llm = Ollama(base_url=os.getenv('OLLAMA_ENDPOINT'), model="phi:2.7b")

    # execute the LLM
    response = llm.invoke("What is Neo4J?")
    print(response)
