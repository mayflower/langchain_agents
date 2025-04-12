# State Management in KI-Anwendungen

---

## Was ist State Management?

### Definition

Die Verwaltung von Zuständen (State) in einer Anwendung über die Zeit und
mehrere Interaktionen hinweg

### Kritische Komponenten in LLM-Anwendungen

- **Konversationsverläufe**
- **Kontextinformationen**
- **Nutzerpräferenzen**
- **Tool-Status und -Ergebnisse**

---

## Herausforderungen im LLM-Kontext

### Kontextfenster-Begrenzung

- GPT-4 Turbo: 128K Token (~96.000 Wörter)
- Claude 3 Opus: 200K Token (~150.000 Wörter)
- Trotz großer Fenster: Effizienz und Kosten beachten

### Anforderungen an modernes State Management

- Effiziente Zustandsspeicherung
- Skalierbarkeit bei vielen Nutzern
- Persistenz über Sessions hinweg
- Datenschutzkonformität

---

## Memory-Typen in LangChain

![Memory-Typen](https://i.imgur.com/mwEQwZX.png)

### Konversations-Buffer

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("Hallo!")
memory.chat_memory.add_ai_message("Wie kann ich helfen?")
```

### Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

window_memory = ConversationBufferWindowMemory(k=5)  # Nur letzte 5 Nachrichten
```

---

## Fortgeschrittene Memory-Konzepte

### Summary Memory

- Erstellt Zusammenfassungen früherer Interaktionen
- Reduziert den Token-Verbrauch erheblich

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
summary_memory = ConversationSummaryMemory(llm=llm)
```

### Entity Memory

- Extrahiert und speichert Informationen über Entitäten
- Ermöglicht personalisierte Interaktionen

```python
from langchain.memory import ConversationEntityMemory

entity_memory = ConversationEntityMemory(llm=llm)
```

---

## Zustandsmanagement in der Praxis

### Konversationskette mit Memory

```python
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response = conversation.predict(
    input="Wie heißt die Hauptstadt von Frankreich?")
```

### Token-Optimierung

1. **Komprimierung**: Ältere Nachrichten zusammenfassen
2. **Filterung**: Irrelevante Informationen entfernen
3. **Priorisierung**: Wichtige Informationen vorhalten

---

## Persistierung von Zuständen

### Datenbank-Integration

```python
from langchain.memory.chat_message_histories import RedisChatMessageHistory

redis_memory = RedisChatMessageHistory(
    session_id="user-123",
    redis_url="redis://localhost:6379"
)
```

### Datenspeicherstrategien

- **In-Memory**: Temporäre Speicherung (Redis, Memcached)
- **Dokument-DB**: Strukturierte Daten (MongoDB, Firestore)
- **Relational**: Komplexe Beziehungen (PostgreSQL)
- **Vector Stores**: Semantische Suche (Pinecone, Weaviate)

---

## Hierarchisches Memory-Management

![Hierarchisches Memory](https://i.imgur.com/JC3XT47.png)

### Drei-Ebenen-Modell

1. **Kurzzeit-Gedächtnis**: Aktuelle Konversation
2. **Mittelfristiges Gedächtnis**: Zusammenfassungen vergangener Interaktionen
3. **Langzeit-Gedächtnis**: Nutzerprofile und persistente Fakten

---

## State Management mit LangGraph

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List


# Zustandstyp definieren
class ConversationState(TypedDict):
    messages: List
    user_info: dict
    current_topic: str


# Graph erstellen
graph = StateGraph(ConversationState)

# Knoten und Kanten definieren
graph.add_node("extract_info", extract_user_info)
graph.add_node("generate_response", generate_response)

graph.add_edge("extract_info", "generate_response")
```

---

## Live Demo: Memory Management

Vergleich verschiedener Memory-Typen:

- Buffer Memory
- Window Memory
- Summary Memory
- Entity Memory

---

## Best Practices

### Architekturempfehlungen

- **Trennung der Zustandstypen**: Verschiedene Speichermechanismen für
  verschiedene Datentypen
- **Modularer Aufbau**: Spezifische Komponenten für verschiedene Aspekte des
  Zustands
- **Skalierbare Infrastruktur**: Auf Wachstum vorbereitet sein

### Datenschutz

- **Datenminimierung**: Nur notwendige Informationen speichern
- **Verschlüsselung**: Sensible Daten schützen
- **Löschkonzepte**: Daten nach Ablauf der Notwendigkeit entfernen

---

## Implementierungsbeispiele

### Zusammenfassungs-basiertes Memory mit Token-Limit

```python
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=400,
    return_messages=True
)
```

### Komplexe Memory-Kombination

```python
from langchain.memory import CombinedMemory

combined_memory = CombinedMemory(
    memories=[
        summary_memory,
        entity_memory
    ]
)
```

---

## Zusammenfassung

### Schlüsselerkenntnisse

- State Management ist essenziell für komplexe KI-Anwendungen
- Verschiedene Memory-Typen für verschiedene Anwendungsfälle
- Hierarchisches Memory und Token-Optimierung für Effizienz
- Persistenz und Skalierbarkeit für Produktionsanwendungen

---

## Fragen?

---

## Praktische Übung

Öffnen Sie das Notebook `06_state_management_notebook.ipynb`, um die
vorgestellten Konzepte selbst auszuprobieren!
