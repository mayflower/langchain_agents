import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.runnables import ConfigurableField
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

load_dotenv()

openai_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"]).configurable_fields(
    temperature=ConfigurableField(id="temperature", is_shared=True)
)
azure_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
).configurable_fields(temperature=ConfigurableField(id="temperature", is_shared=True))
llm = openai_llm.configurable_alternatives(
    ConfigurableField(id="llm"), default_key="openai", azure=azure_llm
)


# Since Embeddings do not implement the Runnable interface, we cannot do the configurable_field trick with it.
# This is just a plain getter.
def embeddings(e: str = "openai"):
    if e == "openai":
        return OpenAIEmbeddings()
    elif e == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"]
        )


def graph_agent_output_printer(data):
    print("\nAgent result: ", data.get("agent_outcome").return_values.get("output"))
