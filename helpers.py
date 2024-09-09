import os
import re
import textwrap
from typing import Any, AsyncIterator, Dict, List, Tuple

import pandas as pd
import torch
from bqplot import ColorScale, LinearScale
from dotenv import load_dotenv
from ipydatagrid import BarRenderer, DataGrid
from langchain._api import suppress_langchain_deprecation_warning
from langchain.evaluation import load_evaluator
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, ensure_config
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langgraph.pregel import ChannelsManager, Pregel
# from langgraph.pregel.algo import prepare_next_tasks
from transformers import AutoModelForMaskedLM, AutoTokenizer
# from langgraph.managed.base import ManagedValueMapping


load_dotenv()


def llm(temperature: float = 0.7, model: str = None, streaming: bool = True, **kwargs):
    """
    wrapper around the OpenAI and Azure OpenAI LLMs. Takes per default the model where the KEY is set in the environment variables.
    First takes OpenAI if the key is set.

    Args:
        temperature (float, optional): temperature for sampling. Defaults to 0.7.
        model (str, optional): model name. Defaults to None. Specify OpenAI model string or Azure deployment name. Otherwise tries to get the model from the environment variables.
    """
    if os.environ.get("OPENAI_API_KEY"):
        model_name = model if model else os.environ.get("OPENAI_MODEL")
        return ChatOpenAI(
            model=model_name, temperature=temperature, streaming=streaming, **kwargs
        )
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        deployment = model if model else os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            temperature=temperature,
            streaming=streaming,
            **kwargs,
        )
    else:
        raise ValueError("No provider secret found in environment variables.")


def embeddings(model=None, **kwargs):
    """
    Returns an instance of the appropriate embeddings provider based on the environment variables.

    Args:
        model (str, optional): The name of the model to use for embeddings. If not provided, the value from the environment variable will be used.
        **kwargs: Additional keyword arguments to be passed to the embeddings provider.

    Returns:
        An instance of the embeddings provider.

    Raises:
        ValueError: If no provider secret is found in the environment variables.
    """
    if os.environ.get("OPENAI_API_KEY"):
        model_name = model if model else os.environ.get("OPENAI_EMBEDDING")
        return OpenAIEmbeddings(model=model_name, **kwargs)
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        deployment = model if model else os.environ.get("AZURE_OPENAI_EMBEDDING_NAME")
        return AzureOpenAIEmbeddings(azure_deployment=deployment, **kwargs)
    else:
        raise ValueError("No provider secret found in environment variables.")


def formatted_output_writer(data):
    """
    Writes the formatted output from the given data.

    Args:
        data (dict): The data containing the agent outcome and return values.

    Returns:
        str: The formatted output.

    """
    return data.get("agent_outcome").return_values.get("output")


def rag_agent_output_streamer(chunks):
    """
    Generator function that processes chunks of output from the RAG agent.

    Args:
        chunks (list): A list of dictionaries representing the output chunks.

    Yields:
        str: The processed output from each chunk.

    """
    for chunk in chunks:
        for key, value in chunk.items():
            if key != "__end__":
                yield f"\nOutput from node '{key}':"
                for message in value.get("messages"):
                    if key == "generate":
                        yield "Generated response:\n"
                        yield "\n".join(textwrap.wrap(message.content, width=120))
                    if key == "retrieve":
                        yield "Retrieved document"
                    if key == "rewrite":
                        yield f"Rewritten question:\n{message.content}"
                    if key == "agent":
                        yield "Agent response:\n"
                        yield (
                            "\n".join(textwrap.wrap(message.content, width=120))
                            if message.content != ""
                            else "No response"
                        )
                        yield "RAG queries:\n"
                        yield " -- ".join(
                            [
                                tool_call.get("function", {}).get("arguments", "")
                                for tool_call in message.additional_kwargs.get(
                                    "tool_calls", []
                                )
                            ]
                        )
                yield "\n---\n"


async def formatted_output_streamer(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """
    Formats the output stream by concatenating the log messages from the input stream.

    Args:
        stream (AsyncIterator[Any]): The input stream containing log messages.

    Yields:
        str: The formatted output log message.

    """
    async for chunk in stream:
        output = ""
        for key, value in chunk.items():
            if key == "agent":
                outcome = value.get("agent_outcome")
                if isinstance(outcome, AgentActionMessageLog):
                    output += f"Agent log:\n\n{outcome.log.strip()}"
                elif isinstance(outcome, AgentFinish):
                    output += f"Agent finished:\n\n{outcome.log.strip()}"
                output += "\n\n----------------------------------------------------------------------------------------\n\n"
            elif key == "action":
                steps: List[Tuple[AgentActionMessageLog, str]] = value.get(
                    "intermediate_steps"
                )
                for index, step in enumerate(steps):
                    output += f"Tool log:\n\n{step[1].strip()}"
                    if index < len(steps) - 1:
                        print("----------------")
                output += "\n\n----------------------------------------------------------------------------------------\n\n"
            elif key == "__end__":
                output = "Done"
        yield output


async def graph_agent_llm_output_streamer_events(app, inputs):
    """
    Stream the output events from the graph agent's language model.

    Args:
        app: The graph agent application.
        inputs: The inputs to be streamed.

    Returns:
        None
    """
    with suppress_langchain_beta_warning():
        async for event in app.astream_events(inputs, version="v1"):
            ev = event["event"]
            if ev == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                function_call_chunk = chunk.tool_calls
                if not function_call_chunk:
                    print(chunk.content, end="", flush=True)


def pretty_print_docs(docs):
    """
    Prints the contents of multiple documents in a pretty format.

    Args:
        docs (list): A list of Document objects.

    Returns:
        None
    """
    print(
        f"\n\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def pretty_print_ranks(corpus, ranks):
    """
    Prints the ranks of the given corpus in a pretty format.

    Args:
        corpus (list): A list of strings representing the corpus.
        ranks (list): A list of dictionaries containing the ranks and corpus IDs.

    Returns:
        None
    """
    print(
        f"\n\n{'-' * 100}\n".join(
            [
                f"Rank {rank['score']:.2f}:\n\n" + corpus[rank["corpus_id"]]
                for rank in ranks
            ]
        )
    )


def pp(text: str) -> List[str]:
    """
    Preprocesses the given text by removing punctuation, converting to lowercase, and splitting into tokens.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        List[str]: A list of tokens after preprocessing.
    """
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    tokens = text.split()
    return tokens


class SpladeEmbeddings:
    """
    A class for generating sparse embeddings using the Splade model.

    Args:
        model_id (str, optional): The identifier of the pre-trained Splade model. Defaults to "naver/splade-cocondenser-ensembledistil".

    Attributes:
        tokenizer: The tokenizer used for tokenizing the input text.
        model: The pre-trained Splade model used for generating embeddings.

    Methods:
        sparse_vector: Generates a sparse vector representation of the input text.
        sparse_tuple: Generates a sparse tuple representation of the input text.
        human_readable_sparse_dict: Generates a human-readable dictionary of token-weight pairs for the input text.
    """

    def __init__(self, model_id: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)

    def sparse_vector(self, text: str) -> torch.Tensor:
        """
        Generates a sparse vector representation of the input text.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: The sparse vector representation of the input text.
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        output = self.model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        return max_val.squeeze()

    def sparse_tuple(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generates a sparse tuple representation of the input text.

        Args:
            text (str): The input text.

        Returns:
            Tuple[List[int], List[float]]: The sparse tuple representation of the input text.
        """
        vector = self.sparse_vector(text)
        indices = vector.nonzero().squeeze().cpu().tolist()
        weights = vector[indices].cpu().tolist()

        return (indices, weights)

    def human_readable_sparse_dict(self, text: str):
        """
        Generates a human-readable dictionary of token-weight pairs for the input text.

        Args:
            text (str): The input text.

        Returns:
            dict: A dictionary containing token-weight pairs, sorted by weight in descending order.
        """
        vector = self.sparse_vector(text)
        cols = vector.nonzero().squeeze().cpu().tolist()
        weights = vector[cols].cpu().tolist()

        idx2token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}
        token_weight_dict = {
            idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
        }
        sorted_token_weight_dict = {
            k: v
            for k, v in sorted(
                token_weight_dict.items(), key=lambda item: item[1], reverse=True
            )
        }

        return sorted_token_weight_dict


def distance_grid(reference, test_set, model=embeddings()):
    """
    Calculate the distance between a reference string and a set of test strings using an embedding model.

    Args:
        reference (str): The reference string.
        test_set (list): A list of test strings.
        model: The embedding model to use (default: embeddings()).

    Returns:
        DataGrid: A DataGrid object containing the test strings and their corresponding distances.

    """
    evaluator = load_evaluator("embedding_distance", embeddings=model)
    distances = []

    with suppress_langchain_deprecation_warning():
        for item in test_set:
            distance = evaluator.evaluate_strings(prediction=item, reference=reference)
            distances.append(distance["score"])

    df = pd.DataFrame({"Wort": test_set, "Entfernung": distances})
    renderers = {
        "Entfernung": BarRenderer(
            horizontal_alignment="center",
            bar_color=ColorScale(min=0, max=1, scheme="viridis"),
            bar_value=LinearScale(min=0, max=1),
        )
    }
    grid = DataGrid(df, base_column_size=250, renderers=renderers)
    grid.transform(
        [
            {"type": "sort", "columnIndex": 2, "desc": False},
        ]
    )
    return grid


def qdr_client():
    """
    Creates and returns a QdrantClient object for interacting with a Qdrant collection.

    Returns:
        client (QdrantClient): The QdrantClient object.
        collection_name (str): The name of the collection.
        vector_name (str): The name of the sparse vector.
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        print("Please install qdrant_client first.")
        return
    client = QdrantClient(":memory:")
    collection_name = "sparse_collection"
    vector_name = "sparse_vector"
    client.create_collection(
        collection_name,
        vectors_config={},
        sparse_vectors_config={
            vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )
    return client, collection_name, vector_name


def is_resumeable(app: Pregel, config: Dict):
    """
    Checks if the application is resumeable.

    Args:
        app (Pregel): The Pregel application.
        config (Dict): The configuration dictionary.

    Returns:
        bool: True if there are tasks to be resumed, False otherwise.
    """
    checkpoint = app.checkpointer.get(config)
    # Handle channels
    with ChannelsManager(app.channels, checkpoint) as channels:
        # Initialize ManagedValueMapping
        managed_values = ManagedValueMapping(app.managed_values_dict)

        # Ensure configuration
        ensured_config = ensure_config(config)

        # Prepare next tasks
        _, tasks = prepare_next_tasks(
            checkpoint,
            app.nodes,
            channels,
            managed_values,
            ensured_config,
            -1,
            for_execution=False,
        )

    return tasks


def interactive_conversation(app: Runnable):
    """
    Run an interactive conversation with the given app.

    Args:
        app (Runnable): The app to interact with.

    Returns:
        None
    """
    while True:
        user = input("Enter message (q to quit): ")
        if user in {"q", "Q"}:
            print("Byebye")
            break
        response = app.invoke(user)
        for k, v in response.items():
            print(f"{k.title()}:")
            print(f"{v}\n")
        print("\n-------------------\n", flush=True)


def interactive_langgraph_conversation(
    app, config={"configurable": {"thread_id": "1"}}, k=0
):
    """
    Run an interactive conversation with the langgraph agent.

    Args:
        app: The langgraph agent application.
        config: Configuration options for the langgraph agent (default: {"configurable": {"thread_id": "1"}}).
        k: The number of previous messages to display in the conversation history (default: 0).

    Returns:
        None
    """
    while True:
        user = input("Enter message (q to quit): ")
        if user in {"q", "Q"}:
            print("Byebye")
            break
        response: List[BaseMessage] = app.invoke(HumanMessage(user), config)
        print("Input:")
        print(f"{response[-2].content}\n")
        print("History:")
        if k >= 1:
            for message in response[-(k * 2 + 2) : -2]:
                print(f"{message.type.title()}: {message.content}")
        else:
            for message in response[0:-2]:
                print(f"{message.type.title()}: {message.content}")
        print("\nResponse")
        print(f"{response[-1].content}\n")

        print("\n-------------------\n", flush=True)
