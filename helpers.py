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

async def graph_agent_llm_output_streamer(app, inputs):
    async for output in app.astream_log(inputs, include_types=["llm"]):
        for op in output.ops:
            if op["path"] == "/streamed_output/-":
                # this is the output from .stream()
                ...
            elif op["path"].startswith("/logs/") and op["path"].endswith(
                "/streamed_output/-"
            ):
                # because we chose to only include LLMs, these are LLM tokens
                token_content = op["value"].content
                if token_content:
                    print(op["value"].content, end="", flush=True)
