# LangGraph

## Überblick

- Funktionsweise und Architektur von LangGraph
- Unterschied zwischen LangChain und LangGraph
- Erstellung von Agenten mit mehreren Werkzeugen
- Praktische Anwendungsbeispiele
- Integration in bestehende Systeme

## Was ist LangGraph?

- Framework zur Erstellung von komplexen, zustandsbasierten Anwendungen mit LLMs
- Basiert auf gerichteten Graphen, die Arbeitsabläufe definieren
- Erweitert LangChain um zustandsbasierte Workflows
- Eignet sich besonders für komplexe, mehrstufige Entscheidungsprozesse
- Ermöglicht präzisere Steuerung des Kontrollflusses als einfache Chains

## Schlüsselkomponenten von LangGraph

- **Knoten (Nodes)**: Verarbeitungseinheiten, die bestimmte Aufgaben ausführen
- **Kanten (Edges)**: Definieren den Fluss zwischen den Knoten
- **Zustand (State)**: Persistente Informationen, die zwischen Knotenaufrufen erhalten bleiben
- **Bedingungen (Conditions)**: Entscheiden, welche Pfade im Graphen genommen werden
- **Entry/Exit Points**: Definieren Start- und Endpunkte des Graphen

## Unterschied zwischen LangChain und LangGraph

| LangChain                      | LangGraph                               |
|--------------------------------|-----------------------------------------|
| Fokus auf Komponenten & Chains | Fokus auf zustandsbasierte Workflows    |
| Lineare Verarbeitungspipelines | Komplexe, verzweigte Prozessabläufe     |
| Einfache Verkettung (a → b → c)| Gerichtete Graphen mit Entscheidungen   |
| Wenig native Zustandsverwaltung| Integrierte Zustandsverwaltung          |
| Gut für einfache Aufgaben      | Ideal für komplexe, iterative Prozesse  |

## Grundlegende Konzepte

### StateGraph vs. MessageGraph

```python
# StateGraph - für vollständige Kontrolle des Zustands
from langgraph.graph import StateGraph
from typing import TypedDict, List

class State(TypedDict):
    messages: List[dict]
    current_step: str

workflow = StateGraph(State)
```

```python
# MessageGraph - vereinfachte Variante für Chatanwendungen
from langgraph.graph import MessageGraph

graph_builder = MessageGraph()
```

## Erstellung eines einfachen Graphen

```python
from langgraph.graph import MessageGraph

# Graph-Builder initialisieren
graph_builder = MessageGraph()

# Knoten hinzufügen
graph_builder.add_node("chatbot_node", model)

# Einstiegspunkt definieren
graph_builder.set_entry_point("chatbot_node")

# Endpunkt definieren
graph_builder.set_finish_point("chatbot_node")

# Kompilieren
simple_graph = graph_builder.compile()
```

## Komplexer Graph mit Verzweigungen

```python
from langgraph.graph import StateGraph
import operator
from typing import Annotated, TypedDict, List

# State-Definition mit Typannotationen
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    tools_used: List[str]

# Workflow definieren
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tool_executor", execute_tools)

# Bedingte Kanten hinzufügen
workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "use_tool": "tool_executor",
        "respond": END,
    },
)
workflow.add_edge("tool_executor", "agent")

# Einstiegspunkt definieren
workflow.set_entry_point("agent")

# Kompilieren
graph = workflow.compile()
```

## Tool-Integration in LangGraph

- Tools können in LangGraph-Knoten integriert werden
- Werkzeuge werden als Funktionen definiert und im Graphen verwendet
- Mehrere Tools können zu einem Agenten kombiniert werden
- Bedingte Logik entscheidet, welches Tool wann verwendet wird

```python
from langchain.tools import tool

@tool
def search_tool(query: str) -> str:
    """Sucht nach Informationen im Internet."""
    # Implementierung der Suchfunktion
    return f"Ergebnisse für: {query}"

@tool
def calculator(expression: str) -> str:
    """Berechnet mathematische Ausdrücke."""
    try:
        return str(eval(expression))
    except:
        return "Konnte den Ausdruck nicht berechnen."

# Tools im Graphen verwenden
tools = [search_tool, calculator]
```

## ReAct-Muster in LangGraph

- **Re**asoning and **Act**ing: Denken und Handeln im Wechsel
- LangGraph bildet ReAct perfekt durch Knotenstruktur ab:
  1. Reasoning-Knoten: Analysiert die Situation und plant
  2. Acting-Knoten: Führt die geplanten Aktionen aus
  3. Observation-Knoten: Sammelt Ergebnisse und Beobachtungen

```python
# Vereinfachte ReAct-Implementierung
workflow = StateGraph(AgentState)
workflow.add_node("reasoning", analyze_and_plan)
workflow.add_node("acting", execute_action)
workflow.add_node("observation", gather_results)

workflow.add_edge("reasoning", "acting")
workflow.add_edge("acting", "observation")
workflow.add_edge("observation", "reasoning")
```

## Praktische Anwendungsbeispiele

1. **Komplexe Entscheidungsbäume**
   - Beratungsgespräche mit mehreren möglichen Pfaden
   - Troubleshooting-Assistenten

2. **Multi-Agenten-Systeme**
   - Teams von spezialisierten Agenten
   - Rollenbasierte Konversationen

3. **Datenverarbeitungspipelines**
   - ETL-Prozesse mit LLM-Einbindung
   - Datentransformationen mit Feedback-Schleifen

4. **Interaktive Tutorials**
   - Schrittweise Anleitungen mit adaptivem Feedback
   - Lernpfade basierend auf Nutzerfähigkeiten

## Fortgeschrittene Techniken

### Parallele Verarbeitung

```python
# Parallele Ausführung von Knoten
workflow.add_node("research", research_function)
workflow.add_node("planning", planning_function)
workflow.add_node("summarize", combine_results)

# Beide Knoten parallel starten
workflow.add_edge("start", "research")
workflow.add_edge("start", "planning")

# Auf Abschluss beider Prozesse warten
workflow.add_edge("research", "summarize")
workflow.add_edge("planning", "summarize")
```

### Zustandstypisierung für robustere Agenten

```python
from typing import TypedDict, List, Dict, Union, Literal
from pydantic import BaseModel

class ToolCall(BaseModel):
    tool_name: str
    tool_input: Dict

class AgentState(TypedDict):
    messages: List[dict]
    current_tool: Union[ToolCall, None]
    status: Literal["thinking", "calling_tool", "finished"]
    intermediate_steps: List[tuple]
```

## Integration in bestehende Systeme

- LangGraph kann als Kernkomponente einer größeren Anwendung dienen
- Integration durch REST-APIs, Webhooks oder direkte Einbindung
- Kann mit Datenbanken, UIs und anderen Services verbunden werden
- Modularität ermöglicht schrittweise Migration bestehender Systeme

## Fehlerbehebung und Debugging

- **Tracing**: Verfolgen des Ausführungspfades durch den Graphen
- **Checkpoints**: Speichern von Zwischenzuständen für Analyse
- **Visualisierung**: Darstellung des Graphen mit Tools wie Graphviz
- **Logs**: Detaillierte Protokollierung der Knotenausführung

## Zusammenfassung

- LangGraph erweitert LangChain um leistungsstarke zustandsbasierte Workflows
- Grafenbasierter Ansatz ermöglicht komplexe, nicht-lineare Prozesse
- Besonders nützlich für:
  - Mehrstufige Entscheidungsprozesse
  - Agenten mit vielen Tools und Fähigkeiten
  - Komplexe Konversationsabläufe mit Zustandsspeicherung
- Skalierbar von einfachen Anwendungen bis zu Enterprise-Lösungen

## Nächste Schritte

- Erkunden der LangGraph-Dokumentation und Beispiele
- Experimentieren mit verschiedenen Graphstrukturen
- Kombinieren mit anderen LangChain-Komponenten
- Aufbau eigener Agenten für spezifische Anwendungsfälle
