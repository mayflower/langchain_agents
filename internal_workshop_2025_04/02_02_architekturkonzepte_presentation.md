# Architekturkonzepte für LLM-Anwendungen

## Übersicht

- Agentenbasierte Ansätze
- ReAct Pattern (Reasoning and Acting)
- Graph-RAG und Hierarchical RAG
- Praktische Anwendungsbeispiele

---

## 1. Agentenbasierte Ansätze

### Was sind LLM-Agenten?

- KI-Systeme, die **eigenständig Entscheidungen treffen** und Aktionen ausführen
- Kombinieren LLMs mit externen **Tools** und einem **Entscheidungsprozess**
- Lösen komplexe, mehrstufige Aufgaben

### Anatomie eines Agenten

1. **LLM-Kern**: Versteht Kontext, plant Schritte, trifft Entscheidungen
2. **Tool-Set**: Spezialisierte Funktionen für bestimmte Aufgaben
3. **Arbeitsspeicher**: Verfolgt Konversationsverlauf und Zwischenergebnisse
4. **Steuerungslogik**: Koordiniert den Workflow zwischen LLM und Tools

---

## Vorteile agentenbasierter Ansätze

### Warum Agenten einsetzen?

- **Autonomie**: Selbstständiges Handeln ohne ständige Anleitung
- **Flexibilität**: Dynamische Auswahl verschiedener Tools je nach Situation
- **Komplexität**: Zerlegung von Problemen in lösbare Teilschritte
- **Erweiterbarkeit**: Einfache Integration neuer Funktionen und Tools

### Beispiel: Einfacher Agent in LangChain

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI

# LLM initialisieren
llm = OpenAI(temperature=0)

# Tools laden
tools = load_tools(["llm-math"], llm=llm)

# Agenten erstellen
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agenten ausführen
agent.invoke("Was ist 15 hoch 0.5?")
```

---

## 2. ReAct Pattern

### Was ist ReAct?

- **Re**asoning + **Act**ion = ReAct
- Ursprung: Forschungspaper "ReAct: Synergizing Reasoning and Acting in Language
  Models"
- Strukturierter Prozess, der **transparentes Denken** mit **gezielten Aktionen
  ** verbindet

### Der ReAct-Prozess

1. **Thought (Gedanke)**: Explizites Nachdenken über das Problem
2. **Action (Aktion)**: Auswahl und Ausführung einer Aktion/eines Tools
3. **Observation (Beobachtung)**: Erfassung der Ergebnisse
4. **Wiederholung**: Bis die Aufgabe gelöst ist

---

## ReAct in der Praxis

### Beispiel: Frage in mehreren Schritten lösen

```
Frage: Wie alt ist Angela Merkel und was ist diese Zahl quadriert?

Thought: Ich muss zuerst das Alter von Angela Merkel herausfinden und dann dieses Alter quadrieren.
Action: Search
Action Input: "Angela Merkel Alter"

Observation: Angela Merkel ist 69 Jahre alt (geboren am 17. Juli 1954).

Thought: Jetzt muss ich 69 quadrieren.
Action: Calculator
Action Input: 69^2

Observation: 4761

Thought: Ich habe beide Informationen. Angela Merkel ist 69 Jahre alt und 69² = 4761.
Final Answer: Angela Merkel ist 69 Jahre alt und diese Zahl quadriert ergibt 4761.
```

---

## Vorteile des ReAct-Musters

### Warum ReAct verwenden?

- **Transparenz**: Der Denkprozess wird sichtbar und nachvollziehbar
- **Selbstkorrektur**: Fehler können erkannt und korrigiert werden
- **Strukturierte Problemlösung**: Systematisches Vorgehen bei komplexen
  Aufgaben
- **Verbesserte Zuverlässigkeit**: Reduziert Halluzinationen durch Faktenprüfung

### Bei welchen Aufgaben ist ReAct besonders effektiv?

- Mehrstufige Recherche-Aufgaben
- Mathematische Problemlösungen
- Aufgaben, die Faktenüberprüfung erfordern
- Interaktionen mit mehreren externen Systemen

---

## 3. Graph-RAG

### Was ist Graph-RAG?

- Erweiterung des klassischen RAG-Ansatzes (Retrieval Augmented Generation)
- Modellierung von Informationen als **Graph-Struktur**
- Berücksichtigung von **Beziehungen** zwischen Informationseinheiten

### Hauptkomponenten von Graph-RAG

- **Knoten**: Dokumente, Abschnitte oder Konzepte
- **Kanten**: Beziehungen zwischen Informationseinheiten
- **Traversierungslogik**: Algorithmen zum Durchlaufen des Graphen

![Graph-RAG Struktur](https://miro.medium.com/v2/resize:fit:1400/1*MKHx5mJQepbRyLvf4i7EcA.png)

---

## Vorteile von Graph-RAG

### Warum Graph-RAG einsetzen?

- **Kontextreichere Informationen**: Verständnis für Zusammenhänge
- **Präzisere Antworten**: Berücksichtigung von Beziehungen zwischen Fakten
- **Bessere Navigation**: Gezielte Exploration verwandter Informationen
- **Ganzheitliches Verständnis**: Erfassung komplexer Wissensstrukturen

### Beispiel: Graph-RAG mit LangGraph

```python
from langgraph.graph import StateGraph

# Graph erstellen
graph = StateGraph(GraphState)

# Knoten hinzufügen
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate_answer)

# Kanten definieren
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

# Graph kompilieren und ausführen
chain = graph.compile()
```

---

## 4. Hierarchical RAG

### Was ist Hierarchical RAG?

- Organisation von Wissen in **hierarchischen Ebenen**
- Schrittweise Verfeinerung der Informationssuche
- Verbessert die Informationsextraktion bei großen Datenmengen

### Typische Hierarchie-Ebenen

1. **Oberste Ebene**: Allgemeine Kategorien, Dokumenttitel, Zusammenfassungen
2. **Mittlere Ebene**: Kapitel, Abschnitte, Themenbereiche
3. **Unterste Ebene**: Detaillierte Textpassagen, spezifische Fakten

![Hierarchical RAG](https://miro.medium.com/v2/resize:fit:1400/1*qOI4V4CdXPQJcU9AswPjAQ.png)

---

## Funktionsweise von Hierarchical RAG

### Der hierarchische Suchprozess

1. **Grobsuche**: Identifikation relevanter Dokumentkategorien
2. **Verfeinerung**: Eingrenzung auf relevante Dokumente/Abschnitte
3. **Detailsuche**: Extraktion spezifischer Informationen

### Vorteile

- **Effizienz**: Reduziert den Suchraum schrittweise
- **Skalierbarkeit**: Geeignet für sehr große Dokumentensammlungen
- **Präzision**: Fokussiert auf die relevantesten Informationen
- **Kontextbewusstsein**: Berücksichtigt größere Zusammenhänge

---

## Anwendungsfälle in der Praxis

### Wo werden diese Architekturen eingesetzt?

- **Kundenservice-Bots**: Autonomous Agents mit ReAct für komplexe Anfragen
- **Forschungsassistenten**: Graph-RAG für wissenschaftliche Literaturrecherche
- **Rechtliche Analysetools**: Hierarchical RAG für Gesetzestexte und
  Präzedenzfälle
- **Medizinische Assistenzsysteme**: Kombination aller Ansätze für klinische
  Entscheidungsunterstützung

### Beispiel: Enterprise-Wissensbasis

- **Level 1**: Produktkategorien und Hauptdokumente (Hierarchical RAG)
- **Level 2**: Verknüpfung verwandter Dokumente (Graph-RAG)
- **Level 3**: Autonome Problemlösung für komplexe Fragen (Agent mit ReAct)

---

## Zusammenfassung und Best Practices

### Wann welche Architektur wählen?

- **Einfache Agenten**: Für klar definierte, wiederkehrende Aufgaben
- **ReAct**: Wenn Transparenz und schrittweise Problemlösung wichtig sind
- **Graph-RAG**: Bei stark vernetztem Wissen mit vielen Querbezügen
- **Hierarchical RAG**: Für große Dokumentenmengen mit klarer Struktur

### Kombinationsmöglichkeiten

- ReAct + Graph-RAG: Transparente Traversierung durch verknüpfte Dokumente
- Hierarchical RAG + Agents: Effiziente Navigation in großen Wissensbasen
- "Multi-Hop RAG": Kombination von Graph-Eigenschaften mit iterativer
  Verfeinerung

---

## Fragen?

---

## Praktische Übung

Öffnen Sie das Notebook `05_architekturkonzepte_notebook.ipynb`, um die
vorgestellten Konzepte selbst auszuprobieren!
