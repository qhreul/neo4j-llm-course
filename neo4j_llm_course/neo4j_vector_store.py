import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

load_dotenv()


class Neo4JConnector:

    def __init__(self):
        """
        Initialize the Neo4J client to access Vector Store
        """
        self.embedding_provider = OllamaEmbeddings(model="nomic-embed-text:latest")

        self.vector_store = None


    def from_index(self, index_name: str, text_property: str, embedding_property: str):
        """
        Initialize the Neo4J client to access Vector Store
        :param index_name: the name of the index to access
        :param text_property: the name of property containing the text
        :param embedding_property: the name of the property containing the embedding
        """
        self.vector_store = Neo4jVector.from_existing_index(
            self.embedding_provider,
            url=os.getenv('NEO4J_ENDPOINT'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD'),
            index_name=index_name,
            embedding_node_property=embedding_property,
            text_node_property=text_property
        )

    def add_documents(self, documents, index_name: str, text_property: str, embedding_property: str):
        """
        Add documents to the vector store
        :param documents: the documents to be added
        :param index_name: the name of the index to access
        :param text_property: the name of property containing the text
        :param embedding_property: the name of the property containing the embedding
        """
        self.vector_store = Neo4jVector.from_documents(
            documents,
            self.embedding_provider,
            url=os.getenv('NEO4J_ENDPOINT'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD'),
            index_name=index_name,
            text_node_property=text_property,
            embedding_node_property=embedding_property,
            create_id_index=True
        )

    def similarity_search(self, query: str):
        """
        Perform a similarity search
        :param query: the query to be executed
        :return: output from the search
        """
        result = self.vector_store.similarity_search(query)

        # debug
        for doc in result:
            print(doc.metadata['title'], '-', doc.page_content)

        return result


if __name__ == '__main__':
    documents = [
        Document(page_content="A movie where aliens land and attack earth.", metadata={"title": "A Alien Encounter"})
    ]
    # create a vector store
    movie_plot_vector = Neo4JConnector()
    movie_plot_vector.add_documents(documents, "myMovieIndex", "text", "embedding")
    movie_plot_vector.similarity_search("A movie where aliens land and attack earth.")

    # initiate the Ollama client
    chat_llm = ChatOllama(
        base_url=os.getenv('OLLAMA_ENDPOINT'),
        model="gemma:7b",
        temperature=0.0
    )

    plot_retriever = RetrievalQA.from_llm(
        llm=chat_llm,
        retriever=movie_plot_vector.vector_store.as_retriever()
    )

    result = plot_retriever.invoke(
        {"query": "A movie where a mission to the moon goes wrong"}
    )
