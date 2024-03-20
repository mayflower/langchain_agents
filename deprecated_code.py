import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.runnables import ConfigurableField

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

# Der Folgende Code-Abschnitt ist nicht wirklich dafür gedacht, sofort verstanden zu werden. Da steckt etwas Langchain-Magie drin.
# Er sollte bestenfalls überflogen oder einfach nicht beachtet werden.
# Man sollte sich aber merken, was er tut, um bei Bedarf hierher zurück zu springen um Code zu kopieren:
# Definiere ein LLM, das zur Laufzeit konfiguriert werden kann (OpenAI, AzureOpenAI, Temperatur....)
# Diese Langchain-Funktionalität wird sich beim Programmieren später einmal sehr nützlich erweisen.
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.runnables import ConfigurableField

openai_llm = ChatOpenAI(model=os.environ["OPENAI_MODEL"]).configurable_fields(
    temperature=ConfigurableField(id="temperature", is_shared=True)
)
azure_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
).configurable_fields(temperature=ConfigurableField(id="temperature", is_shared=True))
llm = openai_llm.configurable_alternatives(
    ConfigurableField(id="llm"), default_key="openai", azure=azure_llm
)

# Der Folgende Code-Abschnitt ist nicht wirklich dafür gedacht, sofort verstanden zu werden. Da steckt eine ganze Menge Langchain-Magie drin.
# Er sollte bestenfalls überflogen oder einfach nicht beachtet werden.
# Man sollte sich merken, was er tut, um bei Bedarf hierher zurück zu springen um Code zu kopieren:
# Baue eine Langchain-Funktion, die eine definierte Konfigurationsschnittstelle zur Laufzeit mit dem Nutzer bietet.
# Die Schnittstelle kümmert sich außerdem um Integrität!
# Hier kann ein Nutzer eine Funktion aufrufen, die vektorisiert und das Embedding-Modell selbst wählen.

from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.runnables import (
    ConfigurableFieldSingleOption,
    RunnableBinding,
    RunnableConfig,
    chain,
)

load_dotenv()


class EmbedText(RunnableBinding):
    embeddings: str

    def __init__(
        self, embeddings: str = "openai", config: RunnableConfig = None, **kwargs
    ):
        @chain
        def _embed_text(text: str):
            if self.embeddings == "openai":
                _embeddings = OpenAIEmbeddings()
            elif self.embeddings == "azure":
                _embeddings = AzureOpenAIEmbeddings(
                    azure_deployment="textembeddingada002"
                )
            return _embeddings.embed_query(text)

        kwargs.pop("bound", None)
        super().__init__(
            embeddings=embeddings, bound=_embed_text, config=config, **kwargs
        )


embed_text = EmbedText().configurable_fields(
    embeddings=ConfigurableFieldSingleOption(
        id="embeddings",
        default="openai",
        options={"openai": "openai", "azure": "azure"},
    )
)

embed_text.config_schema().schema().get("definitions")
