# Tool Calls mit LLMs in LangChain

## Was sind Tool Calls?

- **Definition**: Mechanismus, mit dem LLMs externe Funktionen aufrufen können
- **Anwendung**: Ermöglicht Integration mit externen Diensten und APIs
- **Vorteile**: Erweitert die Möglichkeiten von LLMs durch Zugriff auf aktuelle
  Daten, externe Berechnungen und spezifische Funktionen

---

## Wie funktionieren Tool Calls?

1. **Registrierung**: Tools werden dem LLM bekannt gemacht (mit Beschreibung und
   Parameter-Schema)
2. **Erkennung**: Das LLM erkennt, wann ein Tool hilfreich sein könnte
3. **Aufruf**: Das LLM generiert strukturierte Aufrufe mit den richtigen
   Parametern
4. **Ausführung**: Die Tool-Funktionen werden tatsächlich ausgeführt (außerhalb
   des LLMs)
5. **Rückmeldung**: Die Ergebnisse werden zurück ans LLM gegeben

---

## Tool Calls vs. Klassisches Prompt Engineering

| Tool Calls                                 | Prompt Engineering           |
|--------------------------------------------|------------------------------|
| Dynamische Datenabfrage                    | Statische Informationen      |
| Zugriff auf aktuelle Informationen         | Nur Trainingsdaten verfügbar |
| Präzise Ausführung spezifischer Funktionen | Approximation von Funktionen |
| Klar definierte Aktionsmöglichkeiten       | Freiform-Antworten           |

---

## Implementierung in LangChain

```python
from langchain.tools import tool


@tool
def wetter_abfragen(ort: str):
    """Gibt das aktuelle Wetter für einen Ort zurück."""
    # Hier würde echte API-Abfrage stehen
    return f"In {ort} sind es aktuell 22°C und sonnig."


# Tools als Liste definieren
tools = [wetter_abfragen]

# Tools an das LLM binden
llm_with_tools = llm.bind_tools(tools)
```

---

## Workflow-Integration mit LangGraph

```python
# Vereinfachter Workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)  # LLM-Aufruf
workflow.add_node("action", call_tools)  # Tool-Ausführung
workflow.set_entry_point("agent")

# Zwischen Agent und Tool-Ausführung wechseln
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",  # Tool ausführen
        "end": END,  # Fertig
    },
)
workflow.add_edge("action", "agent")  # Zurück zum Agenten
```

---

## Praktisches Beispiel

```python
@tool
def währungsumrechnung(betrag: float, von_währung: str, zu_währung: str):
    """Rechnet einen Geldbetrag von einer Währung in eine andere um."""
    kurse = {"EUR": 1.0, "USD": 1.08, "GBP": 0.85, "JPY": 163.2}

    if von_währung not in kurse or zu_währung not in kurse:
        return "Währung nicht unterstützt"

    ergebnis = betrag * (kurse[zu_währung] / kurse[von_währung])
    return f"{betrag} {von_währung} = {ergebnis:.2f} {zu_währung}"
```

---

## Anwendungsfälle

- **Informationsabfrage**: Aktuelle Daten, Wetterberichte, Börsenkurse
- **Berechnungen**: Währungsumrechnung, mathematische Operationen
- **Datenbankzugriff**: Abfragen in strukturierten Datenquellen
- **API-Integration**: Verbindung zu externen Diensten (E-Mail, Kalender, CRM)
- **Dokumentenbearbeitung**: Lesen und Schreiben von Dateien

---

## Fortgeschrittene Konzepte

- **Tool-Chaining**: Mehrere Tools in Sequenz ausführen
- **Entscheidungsbäume**: Dynamische Auswahl von Tools basierend auf
  Zwischenergebnissen
- **Validierung**: Überprüfung der LLM-Ausgaben und Parameter-Typen
- **Fallback-Strategien**: Alternativen bei Tool-Fehlern

---

## Praktische Übungen

1. Implementieren Sie ein einfaches Wetter-Tool
2. Erstellen Sie ein Tool zur Währungsumrechnung
3. Kombinieren Sie beide Tools in einem Agenten
4. Erweitern Sie den Agenten um Fehlerbehebung

---

## Fragen & Diskussion
