import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

load_dotenv()


def llm(temperature: float = 0.7):
    if os.environ["OPENAI_API_KEY"]:
        return ChatOpenAI(
            model=os.environ["OPENAI_MODEL"], temperature=temperature, streaming=True
        )
    elif os.environ["AZURE_OPENAI_API_KEY"]:
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=temperature,
            streaming=True,
        )
    else:
        raise ValueError("No provider secret found in environment variables.")


def embeddings():
    if os.environ["OPENAI_API_KEY"]:
        return OpenAIEmbeddings()
    elif os.environ["AZURE_OPENAI_API_KEY"]:
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"]
        )
    else:
        raise ValueError("No provider secret found in environment variables.")


def formatted_output_writer(data):
    return data.get("agent_outcome").return_values.get("output")
