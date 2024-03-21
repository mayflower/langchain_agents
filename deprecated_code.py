import os

from dotenv import load_dotenv
from langchain_core.runnables import ConfigurableField
from langchain_openai import AzureChatOpenAI, ChatOpenAI

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
from langchain_core.runnables import ConfigurableField
from langchain_openai import AzureChatOpenAI, ChatOpenAI

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

from langchain_core.runnables import (
    ConfigurableFieldSingleOption,
    RunnableBinding,
    RunnableConfig,
    chain,
)
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


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

question = (
    "Five houses stand in a row."
    "Each house has a different color."
    "The nationalities of the house residents are different."
    "Each resident prefers a certain drink."
    "Each resident smokes a certain brand of cigarettes."
    "Each resident keeps a certain pet."
    "The colors are blue, yellow, green, red, and white."
    "The nationalities are Danish, German, British, Norwegian, and Swedish."
    "The drinks are Beer, Coffee, Milk, Tea, and Water."
    "The cigarette brands are Dunhill, Marlboro, Pall Mall, Rothmans, and Winfield."
    "The pets are Fish, Dog, Cat, Horse, and Bird."
    "The British lives in the red house."
    "The Swedish keeps a dog."
    "The Danish likes to drink tea."
    "The green house is to the left of the white house."
    "The owner of the green house drinks coffee."
    "The person who smokes Pall Mall keeps a bird."
    "The man who lives in the middle house drinks milk."
    "The owner of the yellow house smokes Dunhill."
    "The Norwegian lives in the first house."
    "The Marlboro smoker lives next to the one who keeps a cat."
    "The man who keeps a horse lives next to the one who smokes Dunhill."
    "The Winfield smoker likes to drink beer."
    "The Norwegian lives next to the blue house."
    "The German smokes Rothmans."
    "If the Marlboro smoker has a neighbor who drinks water, which resident owns the fish?"
)
