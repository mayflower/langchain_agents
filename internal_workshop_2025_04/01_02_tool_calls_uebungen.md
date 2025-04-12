# Tool Calls: Übungen und Lösungen

## Übung 1: Einfaches Mathe-Tool

**Aufgabe**: Erstellen Sie ein Tool, das grundlegende mathematische Operationen
durchführen kann (Addition, Subtraktion, Multiplikation, Division).

**Lösungsansatz**:

```python
from langchain.tools import tool


@tool
def mathe_berechnung(operation: str, zahl1: float, zahl2: float):
    """Führt eine mathematische Berechnung mit zwei Zahlen durch.
    
    Args:
        operation: Die durchzuführende Operation ("addition", "subtraktion", "multiplikation", "division")
        zahl1: Die erste Zahl
        zahl2: Die zweite Zahl
        
    Returns:
        Das Ergebnis der Berechnung
    """
    operation = operation.lower()

    if operation == "addition":
        return f"{zahl1} + {zahl2} = {zahl1 + zahl2}"
    elif operation == "subtraktion":
        return f"{zahl1} - {zahl2} = {zahl1 - zahl2}"
    elif operation == "multiplikation":
        return f"{zahl1} * {zahl2} = {zahl1 * zahl2}"
    elif operation == "division":
        if zahl2 == 0:
            return "Fehler: Division durch Null nicht möglich"
        return f"{zahl1} / {zahl2} = {zahl1 / zahl2}"
    else:
        return f"Fehler: Operation '{operation}' nicht unterstützt. Unterstützte Operationen: addition, subtraktion, multiplikation, division"
```

## Übung 2: Erweiterte Währungsumrechnung

**Aufgabe**: Erweitern Sie das Währungsumrechnungs-Tool um weitere Währungen und
eine Funktion, die die verfügbaren Währungen auflistet.

**Lösungsansatz**:

```python
@tool
def erweiterte_währungsumrechnung(betrag: float, von_währung: str,
                                  zu_währung: str):
    """Rechnet einen Geldbetrag von einer Währung in eine andere um.
    
    Args:
        betrag: Der umzurechnende Geldbetrag
        von_währung: Quellwährung (z.B. "EUR", "USD", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD")
        zu_währung: Zielwährung (dieselben Optionen wie bei von_währung)
        
    Returns:
        Eine Zeichenkette mit dem umgerechneten Betrag
    """
    # Erweiterte Wechselkurse (basierend auf EUR = 1.0)
    kurse = {
        "EUR": 1.0,  # Euro
        "USD": 1.08,  # US-Dollar
        "GBP": 0.85,  # Britisches Pfund
        "JPY": 163.2,  # Japanischer Yen
        "CHF": 0.97,  # Schweizer Franken
        "CNY": 7.85,  # Chinesischer Yuan
        "CAD": 1.47,  # Kanadischer Dollar
        "AUD": 1.64,  # Australischer Dollar
        "INR": 90.2,  # Indische Rupie
        "BRL": 5.42  # Brasilianischer Real
    }

    # Prüfen, ob die Währungen unterstützt werden
    if von_währung not in kurse:
        return f"Fehler: Quellwährung '{von_währung}' wird nicht unterstützt. Verfügbare Währungen: {', '.join(kurse.keys())}"

    if zu_währung not in kurse:
        return f"Fehler: Zielwährung '{zu_währung}' wird nicht unterstützt. Verfügbare Währungen: {', '.join(kurse.keys())}"

    # Umrechnung durchführen
    ergebnis = betrag * (kurse[zu_währung] / kurse[von_währung])

    # Ergebnis formatieren (2 Dezimalstellen, außer bei JPY und INR)
    if zu_währung in ["JPY", "INR"]:
        ergebnis_formatiert = f"{ergebnis:.0f}"
    else:
        ergebnis_formatiert = f"{ergebnis:.2f}"

    # Ergebnis zurückgeben
    return f"{betrag} {von_währung} = {ergebnis_formatiert} {zu_währung}"


@tool
def verfügbare_währungen():
    """Gibt eine Liste aller verfügbaren Währungen für die Umrechnung zurück."""
    kurse = {
        "EUR": "Euro",
        "USD": "US-Dollar",
        "GBP": "Britisches Pfund",
        "JPY": "Japanischer Yen",
        "CHF": "Schweizer Franken",
        "CNY": "Chinesischer Yuan",
        "CAD": "Kanadischer Dollar",
        "AUD": "Australischer Dollar",
        "INR": "Indische Rupie",
        "BRL": "Brasilianischer Real"
    }

    result = "Verfügbare Währungen für die Umrechnung:\n\n"
    for code, name in kurse.items():
        result += f"- {code}: {name}\n"

    return result
```

## Übung 3: Text-Zusammenfassungs-Tool

**Aufgabe**: Erstellen Sie ein Tool, das einen längeren Text mithilfe des LLMs
zusammenfassen kann.

**Lösungsansatz**:

```python
@tool
def text_zusammenfassen(text: str, max_wörter: int = 50):
    """Fasst einen längeren Text kurz zusammen.
    
    Args:
        text: Der zu zusammenfassende Text
        max_wörter: Maximale Anzahl der Wörter in der Zusammenfassung (Standard: 50)
        
    Returns:
        Eine Zusammenfassung des Textes
    """
    from helpers import llm
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser

    # Zusammenfassungs-Prompt erstellen
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"Du bist ein Experte für präzise Textzusammenfassungen. Fasse den folgenden Text in maximal {max_wörter} Wörtern zusammen."),
        ("human", "{input_text}")
    ])

    # Zusammenfassungs-Chain erstellen
    summarize_chain = prompt | llm() | StrOutputParser()

    # Zusammenfassung generieren
    try:
        summary = summarize_chain.invoke({"input_text": text})
        return summary
    except Exception as e:
        return f"Fehler bei der Zusammenfassung: {str(e)}"
```

## Übung 4: Wetterbericht mit Empfehlungen

**Aufgabe**: Erweitern Sie das Wetter-Tool, um basierend auf den
Wetterbedingungen Empfehlungen für Aktivitäten zu geben.

**Lösungsansatz**:

```python
@tool
def wetter_mit_empfehlung(ort: str):
    """Gibt Wetterinformationen für einen Ort zurück und empfiehlt passende Aktivitäten.
    
    Args:
        ort: Der Name der Stadt oder des Ortes
        
    Returns:
        Wetterbericht und Aktivitätsempfehlungen
    """
    # Beispieldaten (in der Realität würde man eine Wetter-API verwenden)
    wetterdaten = {
        "berlin": {"temp": 22, "bedingung": "sonnig",
                   "regen_wahrscheinlichkeit": 0},
        "hamburg": {"temp": 18, "bedingung": "leicht bewölkt",
                    "regen_wahrscheinlichkeit": 10},
        "münchen": {"temp": 20, "bedingung": "vereinzelte Wolken",
                    "regen_wahrscheinlichkeit": 5},
        "köln": {"temp": 21, "bedingung": "wolkenlos",
                 "regen_wahrscheinlichkeit": 0},
        "frankfurt": {"temp": 23, "bedingung": "heiter bis wolkig",
                      "regen_wahrscheinlichkeit": 15},
        "dresden": {"temp": 19, "bedingung": "regnerisch",
                    "regen_wahrscheinlichkeit": 70},
        "düsseldorf": {"temp": 17, "bedingung": "starker Regen",
                       "regen_wahrscheinlichkeit": 90}
    }

    ort_lower = ort.lower()
    if ort_lower not in wetterdaten:
        return f"Keine Wetterdaten für {ort} verfügbar."

    wetter = wetterdaten[ort_lower]

    # Aktivitätsempfehlungen basierend auf Wetter
    aktivitäten = []

    # Temperaturbasierte Empfehlungen
    if wetter["temp"] >= 25:
        aktivitäten.append("Schwimmbad oder Badesee besuchen")
        aktivitäten.append("Eis essen gehen")
    elif wetter["temp"] >= 20:
        aktivitäten.append("Spaziergang im Park")
        aktivitäten.append("Draußen essen gehen")
    elif wetter["temp"] >= 15:
        aktivitäten.append("Leichte Wanderung")
        aktivitäten.append("Radtour")
    else:
        aktivitäten.append("Museum oder Ausstellung besuchen")
        aktivitäten.append("Café-Besuch")

    # Wetterbedingungs-basierte Empfehlungen
    if wetter["regen_wahrscheinlichkeit"] > 50:
        aktivitäten = ["Indoor-Aktivitäten wie Kino oder Museum",
                       "Shopping in Einkaufszentren"]
    elif "wolkenlos" in wetter["bedingung"] or "sonnig" in wetter["bedingung"]:
        aktivitäten.append("Picknick")

    # Ergebnis zusammenstellen
    ergebnis = f"{ort}: {wetter['temp']}°C, {wetter['bedingung']}, Regenwahrscheinlichkeit: {wetter['regen_wahrscheinlichkeit']}%\n\n"
    ergebnis += "Empfohlene Aktivitäten:\n"
    for aktivität in aktivitäten:
        ergebnis += f"- {aktivität}\n"

    return ergebnis
```

## Übung 5: Integration mehrerer Tools in einen LangGraph-Workflow

**Aufgabe**: Erstellen Sie einen LangGraph-Workflow, der mehrere Tools
integriert und sequenziell nutzt.

**Lösungsansatz**:

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

# Alle Tools in einer Liste zusammenfassen
all_tools = [
    personen_info,
    währungsumrechnung,
    wetter_mit_empfehlung,
    mathe_berechnung,
    text_zusammenfassen,
    verfügbare_währungen
]

# LLM mit allen Tools ausstatten
multi_tool_llm = model.bind_tools(all_tools)

# Erweitertes Prompt-Template mit komplexerer Anweisung
advanced_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """Du bist ein fortschrittlicher Assistent mit Zugriff auf verschiedene Werkzeuge.
        Nutze die verfügbaren Werkzeuge, um komplexe Anfragen zu erfüllen.
        Falls nötig, kombiniere mehrere Werkzeuge und integriere die Ergebnisse in deine Antwort.
        Erkläre kurz, welche Werkzeuge du warum verwendet hast."""
    ),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder("intermediate_steps"),
])

# Erweiterte Chain
advanced_chain = advanced_prompt | multi_tool_llm


# State-Definition mit Fortschrittsverfolgung
class AdvancedAgentState(TypedDict):
    input: str
    intermediate_steps: Annotated[
        list[tuple[AgentActionMessageLog, str]], operator.add]
    tools_used: Annotated[
        list[str], operator.add]  # Liste der verwendeten Tools
    answer: str


# Funktion zum Aufrufen des LLM/Agenten
def call_advanced_model(state, config):
    response = advanced_chain.invoke(state, config=config)

    if isinstance(response, AgentFinish):
        return {"answer": response.answer}
    else:
        # Tool-Namen zur Fortschrittsverfolgung hinzufügen
        used_tools = [tool_call["name"] for tool_call in response.tool_calls]
        return {
            "intermediate_steps": [response],
            "tools_used": used_tools
        }


# Tool-Ausführung wie zuvor
def _invoke_advanced_tool(tool_call):
    tool_map = {tool.name: tool for tool in all_tools}
    tool = tool_map[tool_call["name"]]
    return ToolMessage(tool.invoke(tool_call["args"]),
                       tool_call_id=tool_call["id"])


tool_executor = RunnableLambda(_invoke_advanced_tool)


def call_advanced_tools(state):
    last_message = state["intermediate_steps"][-1]
    return {"intermediate_steps": tool_executor.batch(last_message.tool_calls)}


# Erweiterter Graph
advanced_workflow = StateGraph(AdvancedAgentState)
advanced_workflow.add_node("agent", call_advanced_model)
advanced_workflow.add_node("action", call_advanced_tools)
advanced_workflow.set_entry_point("agent")


# Bedingungen wie zuvor
def should_continue(state):
    if not state["intermediate_steps"]:
        return "continue"

    last_step = state["intermediate_steps"][-1]
    if isinstance(last_step, AgentActionMessageLog) and last_step.tool_calls:
        return "continue"
    return "end"


advanced_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
advanced_workflow.add_edge("action", "agent")

# Graph kompilieren
advanced_graph = advanced_workflow.compile()

# Beispielverwendung
komplexe_anfrage = """
Ich bin in München und möchte wissen, ob es sich lohnt, heute draußen spazieren zu gehen.
Außerdem würde ich gerne wissen, wie viel 250 USD in Euro wert sind, da ich etwas Geld wechseln möchte.
"""

ergebnis = advanced_graph.invoke({
    "input": komplexe_anfrage,
    "intermediate_steps": [],
    "tools_used": []
})
```

## Bonus-Übung: Fehlerrobustes Tool

**Aufgabe**: Erstellen Sie ein Tool, das robust mit Fehlern umgeht und
hilfreiche Fehlermeldungen zurückgibt.

**Lösungsansatz**:

```python
@tool
def robuste_web_suche(suchbegriff: str, max_ergebnisse: int = 3):
    """Führt eine Websuche durch und gibt die relevantesten Ergebnisse zurück.
    
    Args:
        suchbegriff: Der zu suchende Begriff oder die Frage
        max_ergebnisse: Maximale Anzahl der zurückzugebenden Ergebnisse (1-5)
        
    Returns:
        Eine Liste der relevantesten Suchergebnisse
    """
    import time
    import random

    # Parameter-Validierung
    if not suchbegriff or len(suchbegriff.strip()) == 0:
        return "Fehler: Bitte geben Sie einen gültigen Suchbegriff ein."

    if not isinstance(max_ergebnisse, int):
        try:
            max_ergebnisse = int(max_ergebnisse)
        except:
            max_ergebnisse = 3

    max_ergebnisse = max(1,
                         min(5, max_ergebnisse))  # Auf Bereich 1-5 beschränken

    # Simulierte Suchergebnisse (in der Realität würde eine echte Suchmaschinen-API verwendet)
    try:
        # Simulierte Verzögerung und möglicher zufälliger Fehler
        time.sleep(1)  # Simulierte Netzwerkverzögerung

        # Simulierter zufälliger Fehler (10% Wahrscheinlichkeit)
        if random.random() < 0.1:
            raise Exception("Simulierter Netzwerkfehler")

        # Beispiel-Suchergebnisse
        alle_ergebnisse = [
            {
                "titel": f"Ergebnis für '{suchbegriff}' - Artikel 1",
                "url": f"https://example.com/article1?q={suchbegriff.replace(' ', '+')}",
                "zusammenfassung": f"Dies ist eine Zusammenfassung zum Thema '{suchbegriff}', die relevante Informationen enthält..."
            },
            {
                "titel": f"Aktuelle Informationen zu {suchbegriff}",
                "url": f"https://infoportal.com/info/{suchbegriff.replace(' ', '-')}",
                "zusammenfassung": f"Hier finden Sie aktuelle und umfassende Informationen zum Thema {suchbegriff}."
            },
            {
                "titel": f"{suchbegriff} - Umfassende Erklärung",
                "url": f"https://wiki.example.org/wiki/{suchbegriff.replace(' ', '_')}",
                "zusammenfassung": f"Eine detaillierte Erklärung zu {suchbegriff} mit Hintergrundinformationen und Beispielen."
            },
            {
                "titel": f"Fragen und Antworten: {suchbegriff}",
                "url": f"https://qa-portal.com/questions/{suchbegriff.replace(' ', '-')}",
                "zusammenfassung": f"Häufig gestellte Fragen und Expertenmeinungen zum Thema {suchbegriff}."
            },
            {
                "titel": f"Neueste Forschung zu {suchbegriff}",
                "url": f"https://research.edu/papers/{suchbegriff.replace(' ', '+')}",
                "zusammenfassung": f"Aktuelle Forschungsergebnisse und wissenschaftliche Erkenntnisse über {suchbegriff}."
            }
        ]

        # Nur die angeforderte Anzahl von Ergebnissen zurückgeben
        ausgewählte_ergebnisse = alle_ergebnisse[:max_ergebnisse]

        # Ergebnisse formatieren
        ergebnis_text = f"Suchergebnisse für '{suchbegriff}':\n\n"
        for i, ergebnis in enumerate(ausgewählte_ergebnisse, 1):
            ergebnis_text += f"{i}. {ergebnis['titel']}\n"
            ergebnis_text += f"   URL: {ergebnis['url']}\n"
            ergebnis_text += f"   {ergebnis['zusammenfassung']}\n\n"

        return ergebnis_text

    except Exception as e:
        # Fehlerbehandlung
        error_message = str(e)
        if "Simulierter Netzwerkfehler" in error_message:
            return "Entschuldigung, es ist ein Netzwerkfehler aufgetreten. Bitte versuchen Sie es später erneut."
        else:
            return f"Bei der Suche ist ein Fehler aufgetreten: {error_message}. Bitte versuchen Sie einen anderen Suchbegriff oder kontaktieren Sie den Support."
```
