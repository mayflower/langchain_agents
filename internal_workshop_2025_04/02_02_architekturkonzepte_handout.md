# Architekturkonzepte für LLM-Anwendungen

## Handout

---

## 1. Agentenbasierte Ansätze

### Definition

Ein **LLM-Agent** ist ein KI-System, das ein Large Language Model mit externen
Werkzeugen (Tools) kombiniert und selbstständig Entscheidungen treffen kann, um
komplexe Aufgaben zu lösen.

### Komponenten eines Agenten

- **LLM-Kern**: Versteht Anfragen, plant Aktionen, generiert Antworten
- **Tools**: Spezialisierte Funktionen (Websuche, Rechner, API-Zugriffe, etc.)
- **Arbeitsspeicher**: Speichert Konversationsverlauf und Zwischenergebnisse
- **Controller/Entscheidungslogik**: Koordiniert den Workflow

### Grundlegender Ablauf

1. Nutzer stellt eine Anfrage
2. Agent analysiert die Anfrage und plant die notwendigen Schritte
3. Agent wählt passende Tools aus und führt diese aus
4. Ergebnisse werden verarbeitet und gegebenenfalls neue Schritte geplant
5. Endgültige Antwort wird dem Nutzer präsentiert

### Code-Beispiel mit LangChain

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

# LLM initialisieren
llm = OpenAI(temperature=0)

# Tools laden
tools = load_tools(["llm-math", "serpapi"], llm=llm)


# Eigenes Tool erstellen
@tool
def wetter_info(ort: str) -> str:
    """Gibt Informationen zum aktuellen Wetter für einen Ort zurück."""
    # Normalerweise würde hier eine API-Anfrage stehen
    return f"In {ort} sind es heute 22°C und sonnig."


# Agenten erstellen und ausführen
agent = initialize_agent(
    tools + [wetter_info],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.invoke("Wie ist das Wetter in Berlin und berechne 25 hoch 0.5")
```

### Anwendungsfälle

- Kundenservice-Bots mit Zugriff auf interne Systeme
- Forschungsassistenten, die Daten analysieren und aufbereiten
- Persönliche Assistenten, die mehrere APIs verbinden

---

## 2. ReAct Pattern (Reasoning and Acting)

### Definition

**ReAct** steht für **Re**asoning and **Act**ing und beschreibt ein Muster, bei
dem ein LLM-Agent sein Denken (Reasoning) explizit mit konkreten Aktionen (
Acting) verknüpft und diesen Prozess schrittweise dokumentiert.

### Der ReAct-Prozess

1. **Thought (Gedanke)**: Agent überlegt explizit, was zu tun ist
2. **Action (Aktion)**: Agent wählt eine spezifische Aktion/Tool
3. **Observation (Beobachtung)**: Agent erhält das Ergebnis der Aktion
4. **Wiederholen**: Diese Schritte werden wiederholt, bis eine Lösung erreicht
   ist

### Beispiel eines ReAct-Ablaufs

```
Frage: In welchem Jahr wurde der Regisseur von "Titanic" geboren und was ist dieses Jahr plus 50?

Thought: Ich muss zuerst herausfinden, wer der Regisseur von "Titanic" ist, dann sein Geburtsjahr recherchieren und schließlich 50 dazu addieren.

Action: Search
Action Input: "Regisseur Titanic Film"

Observation: James Cameron ist der Regisseur von Titanic.

Thought: Jetzt muss ich das Geburtsjahr von James Cameron herausfinden.

Action: Search
Action Input: "James Cameron Geburtsjahr"

Observation: James Cameron wurde am 16. August 1954 geboren.

Thought: James Cameron wurde also 1954 geboren. Ich muss jetzt 1954 + 50 berechnen.

Action: Calculator
Action Input: 1954 + 50

Observation: 2004

Thought: James Cameron wurde 1954 geboren und 1954 + 50 = 2004.

Final Answer: Der Regisseur von "Titanic", James Cameron, wurde im Jahr 1954 geboren, und dieses Jahr plus 50 ergibt 2004.
```

### Implementierung mit LangChain

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# ReAct-Agent initialisieren
react_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Dies ist der ReAct-Agent
    verbose=True
)

react_agent.invoke(
    "Wer ist der aktuelle Bundeskanzler und in welchem Jahr wurde er geboren?")
```

### Vorteile des ReAct-Musters

- **Transparenz**: Der Denkprozess wird sichtbar und nachvollziehbar
- **Selbstkorrektur**: Fehler können erkannt und korrigiert werden
- **Strukturierte Problemlösung**: Komplexe Aufgaben werden in beherrschbare
  Teilschritte zerlegt
- **Bessere Qualität**: Reduziert Halluzinationen durch Faktenprüfung

---

## 3. Graph-RAG

### Definition

**Graph-RAG** erweitert den klassischen RAG-Ansatz (Retrieval Augmented
Generation) durch die Modellierung von Dokumenten und ihren Beziehungen in einer
Graphstruktur.

### Schlüsselkonzepte

- **Knoten (Nodes)**: Repräsentieren Dokumente, Textabschnitte oder Konzepte
- **Kanten (Edges)**: Stellen Beziehungen zwischen Knoten dar (z.B. "gehört
  zu", "widerspricht", "ergänzt")
- **Traversierung**: Navigation durch den Graphen, um relevante Informationen zu
  finden

### Funktionsweise

1. Eine Anfrage wird in einen oder mehrere Ausgangspunkte im Graphen übersetzt
2. Der Graph wird ausgehend von diesen Punkten traversiert
3. Relevante Knoten werden identifiziert und deren Inhalte extrahiert
4. Die gesammelten Informationen werden zur Generierung einer Antwort verwendet

### Vereinfachtes Code-Beispiel

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List


# Zustandsdefinition
class GraphState(TypedDict):
    query: str
    context: List[str]
    answer: str


# Knotenfunktionen definieren
def retrieve_nodes(state):
    # Dokumente aus dem Graphen abrufen
    return {"context": ["relevanter Kontext aus dem Graphen"]}


def generate_answer(state):
    # Antwort basierend auf Kontext generieren
    return {"answer": "generierte Antwort"}


# Graph erstellen
graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve_nodes)
graph.add_node("generate", generate_answer)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

# Graph kompilieren und ausführen
chain = graph.compile()
```

### Vorteile

- **Kontextbewusstsein**: Berücksichtigt Zusammenhänge zwischen Informationen
- **Präzisere Ergebnisse**: Kann widersprüchliche oder sich ergänzende
  Informationen identifizieren
- **Flexibilität**: Ermöglicht verschiedene Traversierungsstrategien je nach
  Anfrage

---

## 4. Hierarchical RAG

### Definition

**Hierarchical RAG** organisiert Dokumente und Wissen in hierarchischen Ebenen,
um die Suche effizient einzugrenzen und relevante Informationen präziser zu
extrahieren.

### Typische Hierarchie-Ebenen

1. **Oberste Ebene**: Allgemeine Kategorien, Dokumenttitel, Zusammenfassungen
2. **Mittlere Ebene**: Kapitel, Abschnitte, Themenbereiche
3. **Unterste Ebene**: Detaillierte Textpassagen, spezifische Fakten

### Prozess

1. **Grobsuche**: Identifikation relevanter Dokumentkategorien oder
   Themenbereiche
2. **Verfeinerung**: Eingrenzung auf relevante Dokumente oder Abschnitte
3. **Detailsuche**: Extraktion der genau benötigten Informationen

### Konzeptionelles Beispiel

```python
def hierarchical_search(query):
    # Stufe 1: Relevante Kategorien/Dokumente identifizieren
    relevant_categories = search_top_level(query)

    # Stufe 2: Relevante Abschnitte innerhalb dieser Dokumente finden
    relevant_sections = search_mid_level(query, relevant_categories)

    # Stufe 3: Detaillierte Informationen extrahieren
    detailed_info = search_detailed_level(query, relevant_sections)

    # Antwort generieren
    return generate_answer(query, detailed_info)
```

### Vorteile

- **Effizienz**: Reduziert den Suchraum schrittweise
- **Skalierbarkeit**: Geeignet für sehr große Dokumentensammlungen
- **Präzision**: Fokussiert auf die relevantesten Informationen
- **Strukturierte Navigation**: Folgt der natürlichen Organisation von Wissen

---

## Kombinationsansätze in der Praxis

### Agentic RAG

Kombination von RAG mit agentenbasierten Ansätzen:

- Agent entscheidet, welche Dokumente relevant sind
- Agent kann Suchanfragen reformulieren oder präzisieren
- Agent kann zwischen verschiedenen Informationsquellen wechseln

### Multi-Hop RAG

Iteratives Retrieval über mehrere Schritte:

1. Initiale Suche basierend auf der ursprünglichen Anfrage
2. Basierend auf ersten Ergebnissen werden neue Suchanfragen generiert
3. Prozess wird wiederholt, um immer präzisere Informationen zu sammeln

### Graph-basiertes hierarchisches RAG

- Verwendet Graphstrukturen für die Navigation durch hierarchische Informationen
- Kombiniert die Vorteile beider Ansätze für komplexe Wissensbasen

---

## Best Practices und Empfehlungen

### Wann welche Architektur wählen?

| Architektur          | Geeignet für                                           | Weniger geeignet für                              |
|----------------------|--------------------------------------------------------|---------------------------------------------------|
| **Einfacher Agent**  | Klar definierte Aufgaben mit festen Tools              | Umgebungen mit unvorhersehbaren Anforderungen     |
| **ReAct**            | Mehrstufige Problemlösung, die Transparenz erfordert   | Einfache Frage-Antwort-Szenarien                  |
| **Graph-RAG**        | Stark vernetzte Informationen mit vielen Querverweisen | Unstrukturierte oder unzusammenhängende Dokumente |
| **Hierarchical RAG** | Große, hierarchisch organisierte Dokumentensammlungen  | Flache Informationssammlungen ohne klare Struktur |

### Implementierungstipps

- Beginnen Sie mit einem einfachen Ansatz und erweitern Sie schrittweise
- Achten Sie auf gute Logging- und Monitoring-Mechanismen
- Testen Sie verschiedene Kombinationen für Ihren speziellen Anwendungsfall
- Evaluieren Sie die Ergebnisse systematisch mit menschlichem Feedback

---

## Ressourcen zum Weiterlesen

- LangChain-Dokumentation zu
  Agenten: [https://python.langchain.com/docs/modules/agents/](https://python.langchain.com/docs/modules/agents/)
- ReAct-Paper: [https://react-lm.github.io/](https://react-lm.github.io/)
-
LangGraph-Dokumentation: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- RAG-Architekturen bei
  LangChain: [https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)
