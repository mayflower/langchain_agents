# 🦜️🏓 LangServe on Docker

## Overview


LangServe helps developers deploy LangChain runnables and chains as a REST API.

This library is integrated with FastAPI and uses pydantic for data validation.

In addition, it provides a client that can be used to call into runnables deployed on a server. A javascript client is available in LangChainJS.

## Prerequisites

- Docker installed on your machine, an existing openai key.

## Getting Started

Use the `LangChain` CLI to bootstrap a `LangServe` project quickly.

To use the langchain CLI make sure that you have a recent version of `langchain-cli` 
installed. You can install it with `pip install -U langchain-cli`.

```sh
langchain app new apps/myapp
cd apps/myapp
```

And start editing server.py.

To serve the new FastAPI app just use 
```sh
langchain serve
```

You can now modify the server.py:

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes


app = FastAPI(
    title="JokeServe",
    description="A joke server that uses GPT-3.5-turbo to generate jokes.",
    version="0.1.0",
)

# Edit this to add the chain you want to add
add_routes(app, 
           ChatOpenAI(model="gpt-3.5-turbo"),
           path="/openai", 
           )

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

add_routes(app,
           prompt | model,
           path="/chain",
           )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```


