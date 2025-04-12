# LangGraph - Handout

## 1. Einführung in LangGraph

LangGraph ist ein Framework zur Erstellung von zustandsbasierten Anwendungen mit Large Language Models (LLMs). Es erweitert LangChain um die Möglichkeit, komplexe, nicht-lineare Workflows zu definieren und zu verwalten.

### Hauptmerkmale von LangGraph:

- Basiert auf gerichteten Graphen mit Knoten und Kanten
- Bietet integrierte Zustandsverwaltung für persistente Daten
- Unterstützt bedingte Pfade und Entscheidungslogik
- Optimiert für iterative, mehrstufige Prozesse
- Ideal für Agentenanwendungen mit verschiedenen Tools

## 2. Grundlegende Komponenten

### StateGraph vs. MessageGraph

LangGraph bietet zwei Haupttypen von Graphen:

- **MessageGraph**: Vereinfachte API für Chatanwendungen
- **StateGraph**: Vollständige Kontrolle über den Zustand mit Typdefinitionen

```python
# Einfacher MessageGraph
from langgraph.graph import MessageGraph

graph_builder = MessageGraph()
graph_builder.add_node("chatbot_node", model)
graph_builder.set_entry_point("chatbot_node")
graph_builder.set_finish_point("chatbot_node")
simple_graph = graph_builder.compile()

# Ausführung des Graphen
response = simple_graph.invoke({"messages": [{"role": "user", "content": "Hallo, wie geht es dir?"}]})
```

```python
# StateGraph mit typisierten Zuständen
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # Liste von Nachrichten, die additiv erweitert wird

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")
graph = workflow.compile()
```

## 3. Grundstruktur eines LangGraph-Agenten

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated
import operator

# 1. Zustandsdefinition
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    tools_used: List[str]

# 2. Funktionen für die Knoten definieren
def call_model(state):
    messages = state["messages"]
    # LLM aufrufen, um Antwort oder Tool-Call zu generieren
    response = llm.invoke(messages)
    return {"messages": [response], "tools_used": state["tools_used"]}

def execute_tools(state):
    messages = state["messages"]
    last_message = messages[-1]
    # Tool-Call ausführen und Ergebnis zurückgeben
    tool_result = execute_tool_call(last_message)
    tools_used = state["tools_used"] + [last_message.tool_name]
    return {"messages": [tool_result], "tools_used": tools_used}

# 3. Entscheidungslogik für bedingte Kanten
def should_use_tool(state):
    last_message = state["messages"][-1]
    if "tool_calls" in last_message:
        return "use_tool"
    else:
        return "respond"

# 4. Graph erstellen
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tool_executor", execute_tools)

# 5. Kanten definieren
workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "use_tool": "tool_executor",
        "respond": END,  # Vordefinierter Endpunkt
    },
)
workflow.add_edge("tool_executor", "agent")

# 6. Einstiegspunkt festlegen
workflow.set_entry_point("agent")

# 7. Graph kompilieren
graph = workflow.compile()

# 8. Graph ausführen
result = graph.invoke({
    "messages": [{"role": "user", "content": "Wie ist das Wetter in Berlin?"}],
    "tools_used": []
})
```

## 4. Tool-Integration

Tools (Werkzeuge) sind Funktionen, die von einem Agenten aufgerufen werden können, um externe Aktionen auszuführen oder Informationen zu beschaffen.

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

# Tools im LLM registrieren
tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)

# Im Knoten verwenden
def agent_with_tools(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

## 5. Erweiterte Graphstrukturen

### Parallele Verarbeitung

```python
# Parallele Ausführung von Knoten
workflow = StateGraph(State)
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

### Kreisläufe im Graphen

```python
# Iterative Verbesserung durch Feedback-Schleifen
workflow = StateGraph(State)
workflow.add_node("generate", generate_content)
workflow.add_node("evaluate", evaluate_content)
workflow.add_node("refine", refine_content)

workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    is_quality_sufficient,
    {
        "sufficient": END,
        "needs_improvement": "refine"
    }
)
workflow.add_edge("refine", "generate")
```

## 6. ReAct-Muster mit LangGraph

Das ReAct-Muster (Reasoning and Acting) ist ein leistungsfähiges Paradigma für KI-Agenten:

```python
# Implementierung des ReAct-Musters
workflow = StateGraph(AgentState)

# Reasoning: Agent analysiert Problem und plant Aktion
workflow.add_node("reasoning", reasoning_function)

# Acting: Ausführung der geplanten Aktion
workflow.add_node("acting", action_function)

# Observation: Sammlung der Ergebnisse
workflow.add_node("observation", observation_function)

# Verbindung der Knoten im ReAct-Zyklus
workflow.add_edge("reasoning", "acting")
workflow.add_edge("acting", "observation")
workflow.add_edge("observation", "reasoning")

# Ausstiegsbedingung, wenn das Problem gelöst ist
workflow.add_conditional_edges(
    "reasoning",
    is_problem_solved,
    {
        "continue": "acting",
        "solved": END
    }
)
```

## 7. Debugging und Überwachung

LangGraph bietet verschiedene Möglichkeiten zum Debugging und zur Überwachung des Graphen:

```python
# Graph mit Tracing kompilieren
graph = workflow.compile(checkpointer=LangchainCheckpointer())

# Ausführung mit Thread-ID für spätere Analyse
thread_id = "thread_123"
result = graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})

# Zugriff auf den Trace
trace = checkpointer.get_trace(thread_id)
```

## 8. Praktische Anwendungsfälle

LangGraph eignet sich besonders für:

- **Komplexe Entscheidungsbäume**: Beratungs- und Support-Systeme
- **Multi-Agenten-Systeme**: Teams von spezialisierten Agenten
- **Datenverarbeitungspipelines**: ETL-Prozesse mit LLM-Unterstützung
- **Interaktive Tutorials**: Adaptive Lernpfade
- **Geschäftsprozessautomatisierung**: Workflow-Automatisierung mit KI

## 9. Integration mit anderen Frameworks

```python
# Integration mit FastAPI
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Graph mit dem Benutzerkontext initialisieren
    result = graph.invoke({
        "messages": request.messages,
        "context": request.context
    })
    return {"response": result["messages"][-1]}
```

## 10. Best Practices

- Definieren Sie klare Zustandstypen mit TypedDict/Pydantic
- Halten Sie Knotenfunktionen klein und fokussiert
- Verwenden Sie bedingte Kanten für komplexe Entscheidungslogik
- Implementieren Sie Fehlerbehandlung in kritischen Knoten
- Nutzen Sie Tracing für das Debugging komplexer Graphen
- Testen Sie Teilgraphen isoliert, bevor Sie sie kombinieren

## Nützliche Ressourcen

- [LangGraph Dokumentation](https://langchain-ai.github.io/langgraph/)
- [LangChain + LangGraph Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangSmith](https://smith.langchain.com/) für Debugging und Monitoring
