# Neo4J & LLM Fundamentals

- [Description](#description)
- [Development](#development)
  - [Environment Variables](#environment-variables)
  - [How to prepare the environment](#how-to-prepare-the-environment) 

## Description <a name="description"></a>
In this course, you will learn how to integrate Neo4j with Generative AI models using [Langchain](https://python.langchain.com/docs/get_started/introduction).

You will learn why graph databases are a reliable option for grounding Large Language Models (LLMs), using Neo4j to provide factual, reliable information to stop the LLM from giving false information, also known as hallucination.

You will use [Langchain](https://python.langchain.com/docs/get_started/introduction) and Python to interact with an LLM and Neo4j. Langchain provides a robust basis for AI application development and comes with Neo4j integrations for Cypher and Vector Indexes.

This implementation leverages [Ollama](https://python.langchain.com/docs/integrations/llms/ollama), although Langchain allows you to work with the LLM of your choice.

## Development <a name="development"></a>

### Environment Variables <a name="environment-variables"></a>
| **Name**               | **Description**                                    | **Default**            |
|------------------------|----------------------------------------------------|------------------------|
| `OLLAMA_ENDPOINT`      | URL for the Ollama endpoint                        | http://localhost:11434 |

### How to prepare the environment <a name="how-to-prepare-the-environment"></a>
* Install dependencies
  ```
  poetry update
  ```