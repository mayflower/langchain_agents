# Tool Calls mit LLMs in LangChain

## Grundlagen der Tool Calls

Tool Calls ermöglichen es Large Language Models (LLMs), externe Funktionen
aufzurufen und so ihre Fähigkeiten über das reine Textgenerieren hinaus zu
erweitern. In LangChain werden Tools als Python-Funktionen implementiert, die
vom LLM erkannt und aufgerufen werden können.

### Vorteile von Tool Calls:

1. **Zugriff auf aktuelle Informationen**: LLMs können Echtzeit-Daten abfragen,
   die nicht in ihren Trainingsdaten enthalten sind
2. **Spezialisierte Funktionalität**: Komplexe Berechnungen oder
   domänenspezifische Logik ausführen
3. **Integration mit externen Systemen**: Verbindung zu Datenbanken, APIs oder
   anderen Diensten
4. **Deterministische Ergebnisse**: Für bestimmte Aufgaben präzisere und
   konsistentere Ergebnisse als reines Text-Prompting

## Implementierung von Tools in LangChain

Der einfachste Weg, ein Tool zu erstellen, ist der `@tool`-Decorator:

```python
from langchain.tools import tool


@tool
def wetter_abfragen(ort: str):
    """Gibt das aktuelle Wetter für einen Ort zurück.
    
    Args:
        ort: Der Name der Stadt oder des Ortes
    
    Returns:
        Eine Beschreibung des aktuellen Wetters
    """
    # Hier würde in einer echten Implementierung eine API-Abfrage stehen
    return f"In {ort} sind es aktuell 22°C und sonnig."
```

Wichtige Komponenten eines Tools:

- **Funktionsname**: Sollte selbsterklärend sein
- **Docstring**: Erklärt dem LLM Zweck und Verwendung des Tools
- **Parameter**: Klar definierte Eingabeparameter mit Typen
- **Rückgabewert**: Was das Tool zurückgibt (wird dem LLM mitgeteilt)

## Integration in LangChain

```python
from helpers import llm

# Tools als Liste definieren
tools = [wetter_abfragen]

# Tools an das LLM binden
llm_with_tools = llm().bind_tools(tools)

# Einfaches Prompt erstellen
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein hilfreicher Assistent, der Informationen bereitstellt."),
    ("human", "{input}")
])

# Chain erstellen
chain = prompt | llm_with_tools
```

## LangGraph-Integration

Für komplexere Anwendungen kann ein Tool-basierter Agent mit LangGraph erstellt
werden:

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph


# State Definition
class AgentState(TypedDict):
    input: str
    intermediate_steps: Annotated[
        list[tuple[AgentActionMessageLog, str]], operator.add]
    answer: str


# Workflow definieren
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tools)
workflow.set_entry_point("agent")

# Kontrollfluss definieren
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",  # Tool ausführen 
        "end": END,  # Fertig
    },
)
workflow.add_edge("action", "agent")

# Graph kompilieren
graph = workflow.compile()
```

## Praxistipps für effektive Tool-Implementierungen

1. **Klare Beschreibungen**: Der Docstring sollte präzise erklären, was das Tool
   tut und wie es zu verwenden ist
2. **Robuste Fehlerbehandlung**: Tools sollten nicht abstürzen, sondern Fehler
   sinnvoll behandeln und zurückgeben
3. **Sinnvolle Parameter**: Gut strukturierte Parameter mit klaren Namen und
   Typen
4. **Rückgabeformat**: Rückgabewerte sollten für das LLM leicht verständlich
   sein
5. **Modularität**: Jedes Tool sollte eine spezifische Aufgabe erfüllen

## Übungsprojekt

Implementieren Sie ein einfaches Währungsumrechnungstool:

```python
@tool
def währungsumrechnung(betrag: float, von_währung: str, zu_währung: str):
    """Rechnet einen Geldbetrag von einer Währung in eine andere um.
    
    Args:
        betrag: Der umzurechnende Geldbetrag
        von_währung: Quellwährung (z.B. "EUR", "USD", "GBP")
        zu_währung: Zielwährung (z.B. "EUR", "USD", "GBP")
        
    Returns:
        Eine Zeichenkette mit dem umgerechneten Betrag
    """
    kurse = {"EUR": 1.0, "USD": 1.08, "GBP": 0.85, "JPY": 163.2}

    if von_währung not in kurse or zu_währung not in kurse:
        return "Währung nicht unterstützt"

    ergebnis = betrag * (kurse[zu_währung] / kurse[von_währung])
    return f"{betrag} {von_währung} = {ergebnis:.2f} {zu_währung}"
```

## Weiterführende Ressourcen

- [LangChain Dokumentation zu Tools](https://python.langchain.com/docs/modules/tools/)
- [LangGraph Dokumentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI Function Calling API](https://platform.openai.com/docs/guides/function-calling)
