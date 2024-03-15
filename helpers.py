import os
import re
from dotenv import load_dotenv
from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureOpenAIEmbeddings,
)
from typing import List, Tuple
from langchain.evaluation import load_evaluator
import pandas as pd
from ipydatagrid import DataGrid, BarRenderer
from bqplot import LinearScale, ColorScale
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import textwrap
from langchain._api import suppress_langchain_deprecation_warning

load_dotenv()


def llm(temperature: float = 0.7, openai_model: str = None):
    if os.environ["OPENAI_API_KEY"]:
        model = openai_model if openai_model else os.environ.get("OPENAI_MODEL")
        return ChatOpenAI(model=model, temperature=temperature, streaming=True)
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


def rag_agent_output_streamer(chunks):
    for chunk in chunks:
        for key, value in chunk.items():
            if key != "__end__":
                yield f"\nOutput from node '{key}':"
                for message in value.get("messages"):
                    if key == "generate":
                        yield "Generated response:\n"
                        yield "\n".join(textwrap.wrap(message.content, width=120))
                    if key == "retrieve":
                        yield f"Retrieved document"
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


# Only streams the final LLM output from the agent. Uses the latest method which is astream_events (in BETA).


async def graph_agent_llm_output_streamer_events(app, inputs):
    async for event in app.astream_events(inputs, version="v1"):
        ev = event["event"]
        if ev == "on_chat_model_stream":
            function_call_chunk = event["data"]["chunk"].additional_kwargs.get(
                "function_calls", None
            )
            if function_call_chunk is None:
                print(event["data"]["chunk"].content, end="", flush=True)


# Only streams the final LLM output from the agent. Uses astream_log which is stable.


async def graph_agent_llm_output_streamer_log(app, inputs):
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
                function_call = op["value"].additional_kwargs.get(
                    "function_calls", None
                )
                if token_content and not function_call:
                    print(op["value"].content, end="", flush=True)


def pretty_print_docs(docs):
    print(
        f"\n\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def pretty_print_ranks(corpus, ranks):
    print(
        f"\n\n{'-' * 100}\n".join(
            [
                f"Rank {rank['score']:.2f}:\n\n" + corpus[rank["corpus_id"]]
                for rank in ranks
            ]
        )
    )


def pp(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    tokens = text.split()
    return tokens


class SpladeEmbeddings:
    def __init__(self, model_id: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)

    def sparse_vector(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt")
        output = self.model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        return max_val.squeeze()

    def sparse_tuple(self, text: str) -> Tuple[List[int], List[float]]:
        vector = self.sparse_vector(text)
        indices = vector.nonzero().squeeze().cpu().tolist()
        weights = vector[indices].cpu().tolist()

        return (indices, weights)

    def human_readable_sparse_dict(self, text: str):
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


def distance_grid(reference, test_set):
    evaluator = load_evaluator("embedding_distance", embeddings=embeddings())
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
