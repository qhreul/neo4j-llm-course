import os
from dotenv import load_dotenv

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.graphs import Neo4jGraph

load_dotenv()

if __name__ == '__main__':
    llm = ChatOllama(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model="gemma:7b"
    )

    graph = Neo4jGraph(
        url=os.getenv('NEO4J_ENDPOINT'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )

    CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
    Convert the user's question based on the schema.
    
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.

    Schema: {schema}
    Question: {question}
    """

    cypher_generation_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["schema", "question"],
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=cypher_generation_prompt,
        verbose=True
    )

    cypher_chain.invoke({"query": "What role did Tom Hanks play in Toy Story?"})
