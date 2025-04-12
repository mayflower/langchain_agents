# Übungen: State Management in KI-Anwendungen

Diese Übungen helfen Ihnen, die verschiedenen Konzepte des State Managements in
LLM-basierten Anwendungen praktisch anzuwenden.

## Übung 1: Verschiedene Memory-Typen vergleichen

### Aufgabe:

Implementieren Sie eine einfache Konversation mit einem LLM unter Verwendung
verschiedener Memory-Typen und vergleichen Sie die Ergebnisse.

### Schritte:

1. Erstellen Sie ein neues Jupyter Notebook oder nutzen Sie ein bestehendes
2. Implementieren Sie die folgenden Memory-Typen:
    - ConversationBufferMemory
    - ConversationBufferWindowMemory (mit k=2)
    - ConversationSummaryMemory
3. Führen Sie mit jedem Memory-Typ dieselbe Konversation durch (mindestens 5
   Interaktionen)
4. Vergleichen Sie, wie die verschiedenen Memory-Typen den Konversationsverlauf
   speichern
5. Analysieren Sie die Vor- und Nachteile jedes Memory-Typs

### Beispiel-Starter-Code:

```python
from langchain.memory import ConversationBufferMemory,

ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# LLM initialisieren
llm = OpenAI(temperature=0.7)

# Verschiedene Memory-Typen erstellen
buffer_memory = ConversationBufferMemory()
window_memory = ConversationBufferWindowMemory(k=2)
summary_memory = ConversationSummaryMemory(llm=llm)

# Konversationsketten erstellen
buffer_chain = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

window_chain = ConversationChain(
    llm=llm,
    memory=window_memory,
    verbose=True
)

summary_chain = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

# Konversation durchführen (für jeden Memory-Typ die gleiche Konversation)
# Beispielkonversation:
# 1. "Hallo, mein Name ist [Ihr Name]."
# 2. "Was ist die Hauptstadt von Frankreich?"
# 3. "Wie ist das Wetter dort üblicherweise im Frühling?"
# 4. "Kannst du mir einige Sehenswürdigkeiten in dieser Stadt empfehlen?"
# 5. "Wie heiße ich noch mal?"

# Führen Sie diese Konversation mit jeder Kette durch und vergleichen Sie die Ergebnisse
```

### Erweiterung:

Nachdem Sie die verschiedenen Memory-Typen verglichen haben:

- Testen Sie, wie sich die Memory-Typen bei längeren Konversationen verhalten
- Experimentieren Sie mit dem Parameter `k` beim Window Memory
- Passen Sie den `max_token_limit` beim ConversationSummaryBufferMemory an

---

## Übung 2: Hierarchisches Memory implementieren

### Aufgabe:

Entwickeln Sie eine einfache Implementierung eines hierarchischen Memory-Systems
mit Kurz- und Langzeitgedächtnis.

### Schritte:

1. Erstellen Sie eine Klasse `HierarchicalMemory`, die zwei Memory-Typen
   kombiniert:
    - Kurzzeit: ConversationBufferWindowMemory für aktuelle Konversation
    - Langzeit: ConversationEntityMemory für wichtige Fakten/Entitäten
2. Implementieren Sie Methoden zum Hinzufügen von Nachrichten und zum Abrufen
   des Gedächtnisinhalts
3. Testen Sie die Implementierung in einer Konversation, die persönliche
   Informationen und Faktenwissen kombiniert

### Beispiel-Starter-Code:

```python
from langchain.memory import ConversationBufferWindowMemory,

ConversationEntityMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# LLM initialisieren
llm = OpenAI(temperature=0.7)


class HierarchicalMemory:
    def __init__(self, llm, window_size=5):
        # Kurzzeit-Gedächtnis (aktuelle Konversation)
        self.short_term = ConversationBufferWindowMemory(k=window_size)

        # Langzeit-Gedächtnis (Entitäten und wichtige Fakten)
        self.long_term = ConversationEntityMemory(llm=llm)

    def add_user_message(self, message):
        # Fügen Sie die Nachricht beiden Gedächtnistypen hinzu
        self.short_term.chat_memory.add_user_message(message)
        self.long_term.chat_memory.add_user_message(message)

    def add_ai_message(self, message):
        # Fügen Sie die Nachricht beiden Gedächtnistypen hinzu
        self.short_term.chat_memory.add_ai_message(message)
        self.long_term.chat_memory.add_ai_message(message)

    def load_memory_variables(self, inputs):
        # Kombinieren Sie die Variablen aus beiden Gedächtnistypen
        short_term_vars = self.short_term.load_memory_variables(inputs)
        long_term_vars = self.long_term.load_memory_variables(inputs)

        # Hier können Sie die Variablen zusammenführen und formatieren
        # TODO: Implementieren Sie die Zusammenführung

        return {
            "history": short_term_vars.get("history", ""),
            "entities": long_term_vars.get("entities", {})
        }

# Testen Sie die hierarchische Memory-Implementierung
# TODO: Implementieren Sie einen Test für die hierarchische Memory-Klasse
```

### Erweiterung:

- Fügen Sie ein mittelfristiges Gedächtnis hinzu (z.B.
  ConversationSummaryMemory)
- Implementieren Sie eine Methode, die automatisch entscheidet, welche
  Informationen aus dem Kurzzeitgedächtnis ins Langzeitgedächtnis übernommen
  werden sollen
- Testen Sie, ob das System persönliche Informationen über mehrere
  Konversationsthemen hinweg behalten kann

---

## Übung 3: Zustand in einer Datenbank persistieren

### Aufgabe:

Implementieren Sie ein System, das den Konversationsverlauf in einer Datenbank
speichert und bei Bedarf wieder abrufen kann.

### Schritte:

1. Wählen Sie eine einfache Datenbank für die Persistierung (z.B. Redis, SQLite
   oder eine In-Memory-Lösung)
2. Implementieren Sie Funktionen zum Speichern und Laden des
   Konversationsverlaufs
3. Integrieren Sie die Persistierung in eine Konversationskette mit LangChain
4. Testen Sie, dass Konversationen über mehrere Programm-Sessions hinweg
   fortgesetzt werden können

### Beispiel-Starter-Code (mit Redis):

```python
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# LLM initialisieren
llm = OpenAI(temperature=0.7)

# Redis-URL (lokale Installation oder Redis-Cloud)
redis_url = "redis://localhost:6379/0"  # Anpassen an Ihre Redis-Instanz


def create_persistent_memory(session_id):
    """Erstellt ein persistentes Memory-Objekt mit Redis-Backend"""
    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url
    )

    return ConversationBufferMemory(
        memory_key="history",
        chat_memory=message_history
    )


def create_conversation(session_id):
    """Erstellt eine Konversationskette mit persistentem Memory"""
    memory = create_persistent_memory(session_id)

    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )


# Beispiel für die Verwendung
session_id = "user123"  # In einer echten Anwendung würde dies für jeden Benutzer eindeutig sein
conversation = create_conversation(session_id)

# Konversation durchführen (erste Session)
response1 = conversation.predict(input="Hallo, ich bin ein neuer Benutzer!")
print(f"Antwort: {response1}")

# In einer realen Anwendung würde das Programm hier beendet und später neu gestartet
print("\n--- Neue Session wird simuliert ---\n")

# Konversation fortsetzen (zweite Session mit demselben session_id)
conversation2 = create_conversation(session_id)
response2 = conversation2.predict(input="Erinnerst du dich an mich?")
print(f"Antwort: {response2}")
```

### Erweiterungen:

- Implementieren Sie eine Funktion zum Löschen alter Konversationen
- Erstellen Sie eine Funktion, die Konversationen nach Benutzer und Datum
  gruppiert
- Implementieren Sie eine simple Backup-Strategie für die gespeicherten
  Konversationen
- Falls Sie keine Redis-Instanz haben, implementieren Sie die Persistierung mit
  SQLite oder einem einfachen JSON-File

---

## Übung 4: Kontext-Optimierung mit Token-Management

### Aufgabe:

Implementieren Sie ein System, das den Konversationskontext optimiert, um
innerhalb eines Token-Limits zu bleiben.

### Schritte:

1. Erstellen Sie eine Funktion, die die Token-Anzahl eines Textes schätzt
2. Implementieren Sie eine Klasse `TokenOptimizedMemory`, die einen
   Konversationsverlauf auf ein Token-Limit beschränkt
3. Nutzen Sie LLM-basierte Komprimierung für zu lange Konversationsverläufe
4. Vergleichen Sie die Leistung mit und ohne Token-Optimierung

### Beispiel-Starter-Code:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
import os
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# LLM initialisieren
llm = OpenAI(temperature=0.7)


def count_tokens(text, model="gpt-3.5-turbo"):
    """Zählt die ungefähre Anzahl von Tokens in einem Text"""
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(text))
    return token_count


class TokenOptimizedMemory(ConversationBufferMemory):
    def __init__(self, max_token_limit=1000, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.max_token_limit = max_token_limit
        self.llm = llm

    def save_context(self, inputs, outputs):
        """Überschreibt save_context, um Tokens zu zählen und zu optimieren"""
        super().save_context(inputs, outputs)

        # Prüfen Sie die aktuelle Token-Anzahl
        memory_variables = super().load_memory_variables({})
        current_history = memory_variables.get("history", "")
        current_tokens = count_tokens(current_history)

        # Wenn über dem Limit, komprimieren
        if current_tokens > self.max_token_limit and self.llm is not None:
            self._compress_history(current_history)

    def _compress_history(self, current_history):
        """Komprimiert die Konversationshistorie mittels LLM"""
        # TODO: Implementieren Sie die Komprimierungslogik
        # Hier könnten Sie das LLM verwenden, um eine kürzere Zusammenfassung zu erstellen
        # und dann die Chat-History zu aktualisieren
        pass

# Testen Sie die Token-optimierte Memory-Implementierung
# TODO: Vervollständigen Sie den Test für die TokenOptimizedMemory-Klasse
```

### Erweiterungen:

- Implementieren Sie verschiedene Komprimierungsstrategien (z.B.
  Zusammenfassung, Filterung, Priorisierung)
- Fügen Sie Markierungen für wichtige Informationen hinzu, die nie komprimiert
  werden sollten
- Erstellen Sie eine Visualisierung des Token-Verbrauchs im Verlauf einer
  längeren Konversation
- Experimentieren Sie mit dynamischen Token-Limits basierend auf der Komplexität
  der Konversation

---

## Übung 5: State Management mit LangGraph

### Aufgabe:

Implementieren Sie einen einfachen Konversations-Agenten mit LangGraph, der
Zustandsinformationen über mehrere Schritte hinweg verwaltet.

### Schritte:

1. Definieren Sie einen Zustandstyp für Ihren Agenten mit TypedDict
2. Erstellen Sie Funktionen für verschiedene Phasen der
   Konversationsverarbeitung
3. Bauen Sie einen Graphen, der diese Funktionen verbindet und den Zustand
   verwaltet
4. Testen Sie den Agenten mit verschiedenen Konversationsszenarien

### Beispiel-Starter-Code:

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Umgebungsvariablen laden
load_dotenv()

# LLM initialisieren
llm = ChatOpenAI(temperature=0.7)


# Zustandstyp definieren
class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    current_topic: str
    next_step: Literal["extract_info", "generate_response", "end"]


# Funktionen zur Zustandsverarbeitung
def extract_info(state: ConversationState) -> ConversationState:
    """Extrahiert Informationen aus der Konversation und aktualisiert das Benutzerprofil"""
    messages = state["messages"]
    user_profile = state["user_profile"].copy()

    # Letzte Benutzernachricht extrahieren
    last_user_message = next((msg["content"] for msg in reversed(messages)
                              if msg["role"] == "user"), "")

    # TODO: Implementieren Sie hier die Informationsextraktion
    # In einer vollständigen Implementierung würden Sie hier das LLM verwenden,
    # um Informationen aus der Nachricht zu extrahieren

    # Beispiel für eine einfache Extraktion
    if "mein Name ist" in last_user_message:
        # Sehr vereinfachte Extraktion, in der Praxis würden Sie NLP/LLM verwenden
        parts = last_user_message.split("mein Name ist")
        if len(parts) > 1:
            name = parts[1].strip().split()[0].rstrip(".,!?")
            user_profile["name"] = name

    return {
        "user_profile": user_profile,
        "next_step": "generate_response"
    }


def generate_response(state: ConversationState) -> ConversationState:
    """Generiert eine Antwort basierend auf dem aktuellen Zustand"""
    messages = state["messages"]
    user_profile = state["user_profile"]

    # Prompt erstellen
    template = ChatPromptTemplate.from_template("""
    Du bist ein freundlicher Assistent. Beantworte die Anfrage des Benutzers.
    
    Über den Benutzer wissen wir:
    {user_profile}
    
    Die letzte Nachricht des Benutzers war:
    {last_message}
    
    Deine Antwort:
    """)

    # Letzte Benutzernachricht extrahieren
    last_user_message = next((msg["content"] for msg in reversed(messages)
                              if msg["role"] == "user"), "")

    # Antwort generieren
    prompt_value = template.format_messages(
        user_profile=str(user_profile),
        last_message=last_user_message
    )

    response = llm.invoke(prompt_value).content

    # Neue Nachricht zur History hinzufügen
    new_messages = messages.copy()
    new_messages.append({"role": "assistant", "content": response})

    return {
        "messages": new_messages,
        "next_step": "end"
    }


def decide_next_step(state: ConversationState) -> Literal[
    "extract_info", "generate_response", "end"]:
    """Entscheidet, was der nächste Schritt sein soll"""
    return state["next_step"]


# Graph erstellen
graph = StateGraph(ConversationState)

# Knoten hinzufügen
graph.add_node("extract_info", extract_info)
graph.add_node("generate_response", generate_response)

# Kanten und Bedingungen definieren
graph.add_conditional_edges(
    "extract_info",
    decide_next_step,
    {
        "generate_response": "generate_response",
        "end": "end"
    }
)

graph.add_conditional_edges(
    "generate_response",
    decide_next_step,
    {
        "extract_info": "extract_info",
        "end": "end"
    }
)

# Startpunkt setzen
graph.set_entry_point("extract_info")

# Graph kompilieren
chain = graph.compile()

# Testen Sie den Graphen
# TODO: Fügen Sie hier Testcode für den Graphen hinzu
```

### Erweiterungen:

- Fügen Sie weitere Knoten für spezifische Aufgaben hinzu (z.B.
  Sentiment-Analyse, Tool-Aufrufe)
- Implementieren Sie eine Kontextfenster-Verwaltung, die zu lange Konversationen
  automatisch komprimiert
- Erweitern Sie das Benutzerprofil um Präferenzen und vergangene Interaktionen
- Fügen Sie einen Knoten für Fehlerbehandlung hinzu, der bei Problemen aktiviert
  wird

---

## Weiterführende Ressourcen

- LangChain-Dokumentation zu
  Memory: [https://python.langchain.com/docs/modules/memory/](https://python.langchain.com/docs/modules/memory/)
- LangGraph-Dokumentation zum
  Zustandsmanagement: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- Artikel "Building Long-Context Applications with
  LLMs": [https://www.pinecone.io/learn/context-window/](https://www.pinecone.io/learn/context-window/)
- Paper "MemGPT: Towards LLMs as Operating
  Systems": [https://arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
