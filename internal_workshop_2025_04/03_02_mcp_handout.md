# Model Context Protocol (MCP) - Handout

## Grundlagen des Kontextmanagements

Das **Model Context Protocol** beschreibt Methoden und Best Practices für das effiziente Management des Kontextes in Gesprächen mit Large Language Models (LLMs). Da alle LLMs ein begrenztes Kontextfenster haben, ist es wichtig, dieses Fenster optimal zu nutzen.

### Wichtige Begriffe

- **Kontextfenster**: Die maximale Anzahl an Tokens, die ein Modell verarbeiten kann
- **Token**: Eine Grundeinheit der Textverarbeitung (etwa 4 Zeichen im Englischen)
- **Prompt**: Die Eingabe an das Modell, die den aktuellen Kontext enthält
- **Memory**: Mechanismus zur Speicherung relevanter Teile der Konversation

## Kontextmanagement-Strategien

### 1. Buffer Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Mein Name ist Hans"}, {"output": "Hallo Hans!"})
memory.save_context({"input": "Ich komme aus Berlin"}, {"output": "Berlin ist eine schöne Stadt."})

# Abrufen des gesamten Kontexts
context = memory.load_memory_variables({})
print(context)
```

### 2. Window Memory
```python
from langchain.memory import ConversationBufferWindowMemory

# Nur die letzten 2 Nachrichten behalten
window_memory = ConversationBufferWindowMemory(k=2)
window_memory.save_context({"input": "Mein Name ist Hans"}, {"output": "Hallo Hans!"})
window_memory.save_context({"input": "Ich komme aus Berlin"}, {"output": "Berlin ist eine schöne Stadt."})
window_memory.save_context({"input": "Was ist mein Name?"}, {"output": "Dein Name ist Hans."})

# Nur die letzten 2 Interaktionen werden gespeichert
context = window_memory.load_memory_variables({})
print(context)
```

### 3. Summary Memory
```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
summary_memory = ConversationSummaryMemory(llm=llm)
summary_memory.save_context({"input": "Mein Name ist Hans"}, {"output": "Hallo Hans!"})
summary_memory.save_context({"input": "Ich komme aus Berlin"}, {"output": "Berlin ist eine schöne Stadt."})
summary_memory.save_context({"input": "Ich arbeite als Ingenieur"}, {"output": "Interessant! Was für ein Ingenieur bist du?"})

# Die Konversation wird zusammengefasst
summary = summary_memory.load_memory_variables({})
print(summary)
```

### 4. Vector Memory
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Embeddings erstellen
embeddings = OpenAIEmbeddings()
# Vektordatenbank initialisieren
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
vector_memory = VectorStoreRetrieverMemory(retriever=retriever)

# Informationen speichern
vector_memory.save_context({"input": "Mein Lieblingsessen ist Pizza"}, 
                           {"output": "Pizza ist ein beliebtes Gericht."})
vector_memory.save_context({"input": "Ich mag auch Pasta"}, 
                           {"output": "Pasta ist auch sehr lecker."})

# Relevante Informationen abrufen
response = vector_memory.load_memory_variables({"prompt": "Was ist mein Lieblingsessen?"})
print(response)
```

## Praktische Umsetzung in LangChain

### Kombination mit ChatModels
```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
llm = ChatOpenAI(temperature=0.7)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response1 = conversation.predict(input="Mein Name ist Thomas und ich interessiere mich für künstliche Intelligenz.")
response2 = conversation.predict(input="Kannst du mir mehr über Machine Learning erklären?")
response3 = conversation.predict(input="Wie war nochmal mein Name?")
```

### Integration in LangGraph
```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Zustandsdefinition mit integriertem Memory
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    memory: ConversationBufferMemory

# Memory initialisieren
memory = ConversationBufferMemory()

# Startzustand definieren
initial_state = {
    "messages": [HumanMessage(content="Hallo! Ich bin Anna.")],
    "memory": memory
}

# Graph aufbauen
workflow = StateGraph(ConversationState)
# ... weitere Graph-Definitionen
```

## Best Practices

1. **Token-Nutzung überwachen**:
   Verwende Tools wie `tiktoken`, um den Token-Verbrauch zu messen und zu optimieren.

2. **Relevante Informationen priorisieren**:
   Nicht alle Teile einer Konversation sind gleich wichtig. Priorisiere Informationen, die den aktuellen Kontext am meisten beeinflussen.

3. **Hierarchische Strategien**:
   Kombiniere verschiedene Memory-Typen für optimale Ergebnisse.

4. **Regelmäßige Zusammenfassung**:
   Fasse lange Konversationen regelmäßig zusammen, um den Tokenverbrauch zu reduzieren.

5. **Externe Wissensspeicher nutzen**:
   Verwende Vektordatenbanken oder andere externe Speicher für Informationen, die nicht ständig im aktiven Kontext sein müssen.
