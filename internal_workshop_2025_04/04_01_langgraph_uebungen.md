# LangGraph - Übungen

## Übung 1: Einfachen Konversationsgraphen erstellen

**Ziel:** Erstellen Sie einen einfachen Konversationsgraphen mit MessageGraph, der auf Benutzereingaben reagiert.

```python
from langgraph.graph import MessageGraph
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

# 1. Erstellen Sie einen LLM-Wrapper
def call_model(state):
    """Knoten, der das LLM aufruft."""
    # Implementieren Sie den Aufruf des Modells
    # Verwenden Sie state["messages"], um die Konversation zu erhalten
    # Geben Sie die neue Nachricht zurück
    pass

# 2. Erstellen Sie einen MessageGraph
# - Fügen Sie einen Knoten für das Modell hinzu
# - Legen Sie Ein- und Ausgangspunkte fest
# - Kompilieren Sie den Graphen

# 3. Testen Sie Ihren Graphen
initial_messages = [
    {"role": "user", "content": "Hallo! Erkläre mir, was ein Graph ist."}
]
# Führen Sie den Graphen aus und geben Sie die Antwort aus
```

**Erweiterung:**
- Fügen Sie einen Systemknoten hinzu, der jeder Anfrage eine Persönlichkeit verleiht
- Implementieren Sie eine Funktion zur Berechnung der Antwortlänge und geben Sie diese aus

## Übung 2: Zustandsbasierter Agent mit Tools

**Ziel:** Erstellen Sie einen Agenten mit StateGraph, der mehrere Tools verwenden kann.

```python
from langgraph.graph import StateGraph
from langchain.tools import tool
from typing import TypedDict, List, Annotated
import operator

# 1. Definieren Sie zwei einfache Tools
@tool
def calculator(expression: str) -> str:
    """Berechnet einen mathematischen Ausdruck."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Fehler bei der Berechnung: {str(e)}"

@tool
def current_date() -> str:
    """Gibt das aktuelle Datum zurück."""
    from datetime import datetime
    return datetime.now().strftime("%d.%m.%Y")

tools = [calculator, current_date]

# 2. Definieren Sie den Zustandstyp
class AgentState(TypedDict):
    # Implementieren Sie den Zustandstyp mit messages und anderen benötigten Feldern
    pass

# 3. Implementieren Sie die Knotenfunktionen
def agent_node(state):
    """Entscheidet, was zu tun ist."""
    # Implementieren Sie die Logik für den Agent-Knoten
    pass

def tool_node(state):
    """Führt das ausgewählte Tool aus."""
    # Implementieren Sie die Logik für den Tool-Knoten
    pass

# 4. Implementieren Sie die Routing-Funktion
def should_use_tool(state):
    """Entscheidet, ob ein Tool verwendet werden soll oder eine Antwort gegeben wird."""
    # Implementieren Sie die Routing-Logik
    pass

# 5. Erstellen Sie den Graphen
# - Definieren Sie die Knoten und Kanten
# - Verwenden Sie bedingte Kanten für die Entscheidungslogik
# - Kompilieren Sie den Graphen

# 6. Testen Sie Ihren Agenten mit verschiedenen Anfragen
test_inputs = [
    "Was ist 123 * 456?",
    "Welches Datum haben wir heute?",
    "Erkläre mir, was ein Graph ist."
]
```

**Erweiterung:**
- Fügen Sie ein Memory-System hinzu, das frühere Tool-Aufrufe speichert
- Implementieren Sie ein Logging-System, das jede Aktion des Agenten aufzeichnet

## Übung 3: ReAct-Muster implementieren

**Ziel:** Implementieren Sie das ReAct-Muster (Reasoning, Acting, Observation) mit LangGraph.

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated, Dict, Optional
import operator

# 1. Definieren Sie den Zustandstyp für das ReAct-Muster
class ReActState(TypedDict):
    messages: Annotated[List[Dict], operator.add]
    reasoning: Optional[str]  # Aktuelle Überlegung
    action: Optional[str]     # Geplante Aktion
    observation: Optional[str] # Beobachtung nach Aktion
    
# 2. Implementieren Sie die drei Kernfunktionen
def reasoning_step(state):
    """Analysiert den aktuellen Zustand und plant die nächste Aktion."""
    # Implementieren Sie die Reasoning-Logik
    pass

def action_step(state):
    """Führt die geplante Aktion aus."""
    # Implementieren Sie die Action-Logik
    pass

def observation_step(state):
    """Sammelt Beobachtungen aus der Aktion."""
    # Implementieren Sie die Observation-Logik
    pass

# 3. Implementieren Sie die Routing-Funktion
def is_task_complete(state):
    """Prüft, ob die Aufgabe abgeschlossen ist."""
    # Implementieren Sie die Abschlussbedingung
    pass

# 4. Erstellen Sie den ReAct-Graphen
# - Definieren Sie die Knoten für jeden Schritt
# - Verbinden Sie die Knoten zum ReAct-Zyklus
# - Fügen Sie eine Abschlussbedingung hinzu
# - Kompilieren Sie den Graphen

# 5. Testen Sie den ReAct-Agenten mit einer komplexen Aufgabe
complex_task = "Finde heraus, wie viel 27*34 ist und dann addiere 15 zum Ergebnis."
```

**Erweiterung:**
- Implementieren Sie eine Visualisierung des ReAct-Prozesses
- Fügen Sie einen "Reflexions"-Schritt hinzu, der die bisherigen Ergebnisse bewertet

## Übung 4: Multi-Agenten-System mit LangGraph

**Ziel:** Erstellen Sie ein System mit mehreren spezialisierten Agenten, die zusammenarbeiten.

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Annotated
import operator

# 1. Definieren Sie spezialisierte Agenten-Rollen
roles = {
    "coordinator": "Du bist der Koordinator. Deine Aufgabe ist es, Anfragen zu analysieren und zu entscheiden, welcher Spezialist am besten antworten kann.",
    "researcher": "Du bist der Rechercheur. Deine Aufgabe ist es, Fakten zu recherchieren und fundierte Informationen zu liefern.",
    "writer": "Du bist der Autor. Deine Aufgabe ist es, Informationen klar und verständlich zu formulieren.",
    "critic": "Du bist der Kritiker. Deine Aufgabe ist es, Inhalte zu überprüfen und Verbesserungsvorschläge zu machen."
}

# 2. Definieren Sie den Teamzustand
class TeamState(TypedDict):
    messages: Annotated[List[Dict], operator.add]
    current_role: str
    intermediate_results: Dict[str, str]

# 3. Implementieren Sie Knotenfunktionen für jeden Agenten
def coordinator_node(state):
    """Koordiniert den Workflow und entscheidet über den nächsten Schritt."""
    # Implementieren Sie die Koordinator-Logik
    pass

def researcher_node(state):
    """Recherchiert Informationen zur Anfrage."""
    # Implementieren Sie die Recherche-Logik
    pass

def writer_node(state):
    """Formuliert eine klare Antwort basierend auf den Rechercheergebnissen."""
    # Implementieren Sie die Autor-Logik
    pass

def critic_node(state):
    """Überprüft und verbessert die Antwort."""
    # Implementieren Sie die Kritiker-Logik
    pass

# 4. Implementieren Sie Routing-Funktionen
def route_to_specialist(state):
    """Entscheidet, welcher Spezialist als Nächstes dran ist."""
    # Implementieren Sie die Routing-Logik
    pass

def is_response_complete(state):
    """Prüft, ob die Antwort fertig ist."""
    # Implementieren Sie die Abschlussbedingung
    pass

# 5. Erstellen Sie den Team-Graphen
# - Fügen Sie Knoten für jeden Agenten hinzu
# - Definieren Sie den Workflow mit bedingten Kanten
# - Kompilieren Sie den Graphen

# 6. Testen Sie das Multi-Agenten-System
query = "Erkläre die Vorteile und Nachteile von erneuerbaren Energien im Vergleich zu fossilen Brennstoffen."
```

**Erweiterung:**
- Implementieren Sie ein Abstimmungssystem für Entscheidungen
- Fügen Sie einen Feedback-Mechanismus hinzu, der die Teamleistung verbessert

## Übung 5: Persistenter Graph mit Speicherung

**Ziel:** Erstellen Sie einen persistenten Graphen, der Konversationen über mehrere Durchläufe hinweg fortsetzen kann.

```python
from langgraph.graph import StateGraph
from langchain.memory import ChatMessageHistory
from typing import TypedDict, Dict, List, Annotated
import operator
import uuid

# 1. Definieren Sie einen persistenten Zustandstyp
class PersistentState(TypedDict):
    messages: Annotated[List[Dict], operator.add]
    session_id: str
    memory: Dict[str, List[Dict]]  # Speicher für verschiedene Sessions

# 2. Implementieren Sie Funktionen für Zustandsmanagement
def initialize_session(state, session_id=None):
    """Initialisiert eine neue Session oder lädt eine existierende."""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Implementieren Sie die Initialisierungslogik
    pass

def save_state(state):
    """Speichert den aktuellen Zustand in den Speicher."""
    # Implementieren Sie die Speicherlogik
    pass

def load_state(session_id):
    """Lädt einen Zustand aus dem Speicher."""
    # Implementieren Sie die Ladelogik
    pass

# 3. Implementieren Sie den Hauptknoten
def conversation_node(state):
    """Verarbeitet die Konversation und aktualisiert den Speicher."""
    # Implementieren Sie die Konversationslogik
    pass

# 4. Erstellen Sie den persistenten Graphen
# - Definieren Sie die benötigten Knoten
# - Verbinden Sie sie zu einem Workflow
# - Kompilieren Sie den Graphen

# 5. Testen Sie die Persistenz mit mehreren Aufrufen
session_id = str(uuid.uuid4())
conversations = [
    "Hallo! Mein Name ist Max.",
    "Was kannst du dir über mich merken?",
    "Hast du meinen Namen gespeichert?"
]

# Führen Sie mehrere Konversationen mit demselben session_id durch
```

**Erweiterung:**
- Implementieren Sie ein Backup-System für den Fall von Fehlern
- Fügen Sie Zeitstempel und Metadaten zu den gespeicherten Zuständen hinzu

## Bonus-Übung: Graph-Visualisierung

**Ziel:** Erstellen Sie eine Visualisierung eines LangGraph-Workflows.

```python
import graphviz
from langgraph.graph import StateGraph

# 1. Erstellen Sie einen komplexen Graphen (z.B. aus einer der vorherigen Übungen)

# 2. Implementieren Sie eine Funktion zur Visualisierung des Graphen
def visualize_graph(graph, filename="graph_visualization"):
    """Erstellt eine visuelle Darstellung des Graphen mit Graphviz."""
    dot = graphviz.Digraph(comment="LangGraph Workflow")
    
    # Implementieren Sie die Visualisierungslogik
    # - Knoten hinzufügen
    # - Kanten hinzufügen
    # - Spezielle Formatierung für verschiedene Knotentypen
    
    # Speichern und Anzeigen
    dot.render(filename, format="png", cleanup=True)
    return dot

# 3. Visualisieren Sie Ihren Graphen
graph_viz = visualize_graph(your_graph)

# 4. Optional: Implementieren Sie eine Animation des Durchlaufs
def animate_graph_execution(graph, input_state, steps=10):
    """Erstellt eine Schritt-für-Schritt-Visualisierung der Graphenausführung."""
    # Implementieren Sie die Animationslogik
    pass
```

**Erweiterung:**
- Fügen Sie Interaktivität zur Visualisierung hinzu (z.B. mit Jupyter Widgets)
- Implementieren Sie eine Funktion zur Analyse des Graphen (z.B. Pfadlänge, Zyklenerkennung)
