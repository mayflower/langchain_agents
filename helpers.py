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
