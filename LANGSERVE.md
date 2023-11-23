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

To be able to deploy the application with a key on it's own we use dotenv.
```sh
poetry add python-dotenv
```

Create a .env file with the setup of your azure openai endpoint: 
```sh
AZURE_OPENAI_ENDPOINT=sk-your-azure-openai-endpoint-here
AZURE_OPENAI_API_KEY=sk-your-azure-openai-key-here
OPENAI_API_VERSION='2023-05-15'
OPENAI_API_TYPE='azure'
```

You can now modify the server.py:

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI

app = FastAPI(
    title="JokeServe",
    description="A joke server that uses GPT-3.5-turbo to generate jokes.",
    version="0.1.0",
)

# openai transparent default route to openai
add_routes(app, 
           AzureChatOpenAI(
                deployment_name="GPT4",
           ),
            path="/openai",
           )

model = AzureChatOpenAI(
    deployment_name="GPT4",
)

# a joke telling route
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

add_routes(app,
           prompt | model,
           path="/chain",
           )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## Dockerize your service 

See [our example Dockerfile](/services/defaults/Dockerfile) and [example docker-compose.yaml](/services/defaults/docker-compose.yml).

## Docker example service 

We implemented a simple joke service based on the langserve example.

```sh
cd services/jokeservice
docker-compose build
docker-compose up
```
You can now use the following urls to interact with the new service: 

| Description | URL |
|-------------|-----|
|API Documentation|[http://localhost:8001/docs](http://localhost:8001/docs)|
|Swagger Json | [http://localhost:8001/openapi.json](http://localhost:8001/openapi.json)|
|Chain Playground|[http://localhost:8001/chain/playground/](http://localhost:8001/chain/playground/)|

## Use your service with kubernetes 

You can deploy these services on kubernetes using [tilt.dev]("https://tilt.dev/") and [k3d]("https://k3d.io/") for local kubernetes development. 

Please install tilt and k3d locally. 

To use tilt with k3s use the following commands to start a local, docker-based kubernetes cluster and to start a local demo service based on langserve:
```sh
cd services 

k3d cluster create langchain --registry-create registry:5000

# wait around 2 minutes for kubernetes to boot

tilt up
```





