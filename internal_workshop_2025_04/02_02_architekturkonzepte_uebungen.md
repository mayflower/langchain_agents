# Übungen: Architekturkonzepte für LLM-Anwendungen

Diese Übungen helfen Ihnen, die verschiedenen Architekturkonzepte für
LLM-Anwendungen praktisch anzuwenden. Alle Übungen können im Jupyter Notebook
`05_architekturkonzepte_notebook.ipynb` durchgeführt werden.

## Übung 1: ReAct-Agent erweitern

### Aufgabe:

Erweitern Sie den ReAct-Agenten aus dem Notebook um ein eigenes Tool, das einen
praktischen Nutzen hat.

### Schritte:

1. Erstellen Sie eine neue Tool-Funktion mit dem `@tool`-Decorator
2. Die Funktion sollte einen der folgenden Dienste anbieten (oder einen
   eigenen):
    - Wetter-Informationen für einen Ort abrufen (z.B. mit einer Dummy-Funktion)
    - Währungsumrechnung zwischen zwei Währungen
    - Textübersetzung in eine andere Sprache
    - Zusammenfassung eines langen Textes
3. Fügen Sie das Tool zur Tool-Liste des Agenten hinzu
4. Testen Sie den Agenten mit einer Anfrage, die Ihr neues Tool verwendet

### Beispiel-Starter-Code:

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI
from langchain.tools import tool

# LLM initialisieren
llm = OpenAI(temperature=0)

# Basis-Tools laden
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Ihr eigenes Tool hier definieren
@tool
def waehrungsumrechnung(betrag_währung: str) -> str:
    """
    Rechnet einen Geldbetrag von einer Währung in eine andere um.
    Das Eingabeformat muss sein: "100 EUR zu USD" oder "50 USD zu JPY".
    """
    # Hier Ihre Implementierung einfügen
    # Dies ist eine Dummy-Funktion - in der Praxis würden Sie eine externe API verwenden
    try:
        teile = betrag_währung.split()
        betrag = float(teile[0])
        von_waehrung = teile[1]
        zu_waehrung = teile[3]

        # Beispielhafte Wechselkurse (stark vereinfacht)
        kurse = {
            "EUR_USD": 1.1,
            "USD_EUR": 0.91,
            "EUR_GBP": 0.85,
            "GBP_EUR": 1.18,
            "USD_JPY": 150.2,
            # Weitere Kurse hier...
        }

        kurs_key = f"{von_waehrung}_{zu_waehrung}"
        if kurs_key in kurse:
            ergebnis = betrag * kurse[kurs_key]
            return f"{betrag} {von_waehrung} entspricht {ergebnis:.2f} {zu_waehrung}"
        else:
            return f"Wechselkurs für {von_waehrung} zu {zu_waehrung} nicht verfügbar."
    except Exception as e:
        return f"Fehler bei der Umrechnung: {str(e)}"


# Tool zur Liste hinzufügen
tools.append(waehrungsumrechnung)

# Agent initialisieren
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent testen
agent.invoke("Ich habe 500 EUR und möchte wissen, wie viel das in USD ist. " +
             "Außerdem möchte ich wissen, wie viel 1000 USD⁴ ist.")
```

### Erweiterung:

Nachdem Ihr Tool funktioniert, versuchen Sie, es zu verbessern:

- Fügen Sie bessere Fehlerbehandlung hinzu
- Verbessern Sie die Dokumentation des Tools
- Erweitern Sie die Funktionalität (z.B. mehr Währungen oder genauere Daten)

---

## Übung 2: Einen LangGraph mit Feedback-Schleife erstellen

### Aufgabe:

Erweitern Sie den Graph-RAG-Ansatz aus dem Notebook um eine Feedback-Schleife,
die die Qualität der Antwort verbessert.

### Schritte:

1. Starten Sie mit dem Graph-RAG-Beispiel aus dem Notebook
2. Fügen Sie einen neuen Knoten namens `evaluate_answer` hinzu
3. Dieser Knoten sollte die generierte Antwort bewerten und entscheiden, ob:
    - Die Antwort ausreichend ist (→ Fertig)
    - Die Antwort verbesserungswürdig ist (→ Zurück zum Retrieval oder zur
      Generierung)
4. Implementieren Sie die entsprechenden Bedingungen und Kanten im Graphen
5. Testen Sie den erweiterten Graphen mit einer komplexen Anfrage

### Beispiel-Starter-Code:

```python
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser


# Definieren eines Zustands für unseren Graphen mit Feedback
class GraphStateFeedback(TypedDict):
    query: str
    context: List[str]
    answer: str
    feedback: str
    iterations: int


# Embeddings und LLM initialisieren
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# Beispieldaten für die Vektordatenbank
sample_texts = [
    "Berlin ist die Hauptstadt von Deutschland.",
    "München ist die Hauptstadt von Bayern.",
    "Hamburg ist die zweitgrößte Stadt Deutschlands.",
    "Frankfurt ist ein wichtiges Finanzzentrum in Europa.",
    "Köln ist bekannt für seinen Dom."
]

# Vektordatenbank erstellen
vectorstore = Chroma.from_texts(sample_texts, embeddings)


# Knoten-Funktionen definieren
def retrieve(state: GraphStateFeedback) -> GraphStateFeedback:
    """Dokumente aus der Vektordatenbank abrufen"""
    query = state["query"]
    docs = vectorstore.similarity_search(query, k=2)
    return {"context": [doc.page_content for doc in docs]}


def generate_answer(state: GraphStateFeedback) -> GraphStateFeedback:
    """Antwort basierend auf dem Kontext generieren"""
    query = state["query"]
    context = state["context"]

    prompt = ChatPromptTemplate.from_template(
        """Du bist ein hilfreicher Assistent. 
        Verwende den folgenden Kontext, um die Frage zu beantworten.
        
        Kontext: {context}
        
        Frage: {query}
        """
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": "\n".join(context), "query": query})

    return {"answer": answer}


def evaluate_answer(state: GraphStateFeedback) -> GraphStateFeedback:
    """Bewerte die Antwort und gib Feedback"""
    query = state["query"]
    answer = state["answer"]

    prompt = ChatPromptTemplate.from_template(
        """Bewerte die folgende Antwort auf die gegebene Frage.
        
        Frage: {query}
        Antwort: {answer}
        
        Gib konstruktives Feedback, wie die Antwort verbessert werden könnte.
        Bewerte auch, ob die Antwort bereits ausreichend ist oder verbessert werden sollte.
        """
    )

    chain = prompt | llm | StrOutputParser()
    feedback = chain.invoke({"query": query, "answer": answer})

    # Zähle die Iteration
    iterations = state.get("iterations", 0) + 1

    return {"feedback": feedback, "iterations": iterations}


def should_continue(state: GraphStateFeedback) -> Literal["continue", "finish"]:
    """Entscheide, ob der Graph fortgesetzt oder beendet werden soll"""
    feedback = state["feedback"]
    iterations = state["iterations"]

    # Maximale Iterationen begrenzen
    if iterations >= 3:
        return "finish"

    # Prüfen, ob das Feedback auf eine Verbesserung hindeutet
    if "verbessern" in feedback.lower() or "unvollständig" in feedback.lower():
        return "continue"
    else:
        return "finish"


# Graph erstellen
graph = StateGraph(GraphStateFeedback)

# Knoten hinzufügen
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate_answer)
graph.add_node("evaluate", evaluate_answer)

# Startpunkt festlegen
graph.set_entry_point("retrieve")

# Kanten und bedingte Kanten hinzufügen
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "continue": "retrieve",
        "finish": None  # None bedeutet, dass der Graph hier endet
    }
)

# Graph kompilieren
chain = graph.compile()

# Ihre Aufgabe: Vervollständigen Sie den Graphen und testen Sie ihn mit einer komplexen Anfrage
# Tipp: Fügen Sie Code hinzu, um den Ablauf und die Iterationen zu überwachen
```

### Erweiterung:

- Implementieren Sie eine detailliertere Bewertungsfunktion
- Fügen Sie verschiedene Verbesserungspfade hinzu (z.B. zurück zum Retrieval
  oder direkt zur Neugenerierung)
- Experimentieren Sie mit unterschiedlichen Abbruchkriterien

---

## Übung 3: Hierarchical RAG für ein Fachgebiet entwerfen

### Aufgabe:

Entwerfen Sie eine hierarchische RAG-Struktur für ein spezielles Fachgebiet
Ihrer Wahl (z.B. Medizin, Recht, Technik).

### Schritte:

1. Wählen Sie ein Fachgebiet
2. Definieren Sie mindestens drei Hierarchieebenen für dieses Fachgebiet
3. Erstellen Sie eine einfache Beispieldatenstruktur
4. Implementieren Sie den hierarchischen Suchprozess
5. Testen Sie Ihre Implementierung mit einigen Beispielfragen

### Beispiel-Starter-Code für rechtliche Dokumente:

```python
# Beispiel-Hierarchie für rechtliche Dokumente
legal_hierarchy = {
    "level1": [
        {"id": "civil", "title": "Zivilrecht",
         "summary": "Alle zivilrechtlichen Dokumente und Gesetze"},
        {"id": "criminal", "title": "Strafrecht",
         "summary": "Strafrecht und relevante Verfahren"},
        {"id": "admin", "title": "Verwaltungsrecht",
         "summary": "Verwaltungsrechtliche Bestimmungen"}
    ],
    "level2": {
        "civil": [
            {"id": "civil_code", "title": "Bürgerliches Gesetzbuch (BGB)"},
            {"id": "commercial", "title": "Handelsrecht"},
            {"id": "family", "title": "Familienrecht"}
        ],
        "criminal": [
            {"id": "criminal_code", "title": "Strafgesetzbuch (StGB)"},
            {"id": "procedure", "title": "Strafprozessordnung (StPO)"}
        ],
        "admin": [
            {"id": "admin_procedure", "title": "Verwaltungsverfahrensgesetz"},
            {"id": "tax", "title": "Steuerrecht"}
        ]
    },
    "level3": {
        "civil_code": [
            {"id": "bgb_1", "title": "BGB §1: Rechtsfähigkeit",
             "content": "Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung der Geburt."},
            {"id": "bgb_2", "title": "BGB §2: Eintritt der Volljährigkeit",
             "content": "Die Volljährigkeit tritt mit der Vollendung des 18. Lebensjahres ein."}
            # Weitere Paragraphen...
        ],
        "criminal_code": [
            {"id": "stgb_1", "title": "StGB §1: Keine Strafe ohne Gesetz",
             "content": "Eine Tat kann nur bestraft werden, wenn die Strafbarkeit gesetzlich bestimmt war, bevor die Tat begangen wurde."}
            # Weitere Paragraphen...
        ]
        # Weitere Unterkategorien...
    }
}

# Ihre Aufgabe: Implementieren Sie die hierarchische Suche
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(temperature=0)


def hierarchical_legal_search(query: str) -> str:
    """Führt eine hierarchische Suche in rechtlichen Dokumenten durch"""

    # 1. Schritt: Auswahl relevanter Rechtsgebiete (Level 1)
    # Implementieren Sie die Auswahl des relevanten Rechtsgebiets basierend auf der Anfrage

    # 2. Schritt: Auswahl relevanter Gesetze (Level 2)
    # Implementieren Sie die Auswahl des relevanten Gesetzbuchs

    # 3. Schritt: Auswahl spezifischer Paragraphen (Level 3)
    # Implementieren Sie die Auswahl der relevanten Paragraphen

    # 4. Schritt: Generierung einer Antwort basierend auf den gefundenen Informationen
    # Implementieren Sie die Antwortgenerierung

    return "Implementieren Sie die hierarchische Suche"


# Testen Sie Ihre Implementierung
result = hierarchical_legal_search("Was besagt das BGB zur Volljährigkeit?")
print(result)
```

### Erweiterungen:

- Fügen Sie eine vierte Hierarchieebene hinzu (z.B. Gerichtsentscheidungen zu
  bestimmten Paragraphen)
- Implementieren Sie eine Relevanz-Scoring-Funktion für jede Ebene
- Kombinieren Sie die hierarchische Suche mit einer Graphstruktur, um
  Beziehungen zwischen verschiedenen Rechtsbereichen zu modellieren

---

## Übung 4: Agentic RAG implementieren

### Aufgabe:

Implementieren Sie einen Agenten, der RAG-Operationen steuert und optimiert.

### Schritte:

1. Erstellen Sie einen Agenten mit mindestens drei Tools:
    - Ein Tool zum Suchen in der Vektordatenbank
    - Ein Tool zum Umformulieren der Suchanfrage
    - Ein Tool zum Generieren von Antworten
2. Der Agent sollte entscheiden können, wann er welches Tool einsetzt
3. Implementieren Sie eine Strategie zur Verbesserung der Suchergebnisse

### Beispiel-Starter-Code:

```python
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# LLM und Embeddings initialisieren
llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# Beispieldaten für die Vektordatenbank
sample_texts = [
    "Berlin ist die Hauptstadt von Deutschland und hat etwa 3,7 Millionen Einwohner.",
    "München ist die Hauptstadt von Bayern und bekannt für das Oktoberfest.",
    "Hamburg ist die zweitgrößte Stadt Deutschlands und ein wichtiger Hafen.",
    "Frankfurt ist ein bedeutendes Finanzzentrum in Europa und Sitz der Europäischen Zentralbank.",
    "Köln ist bekannt für seinen Dom, der über 600 Jahre gebaut wurde.",
    "Stuttgart ist ein wichtiger Automobilstandort und Heimat von Mercedes-Benz und Porsche."
]

# Vektordatenbank erstellen
vectorstore = Chroma.from_texts(sample_texts, embeddings)


# Tools definieren
@tool
def suche_in_dokumenten(query: str) -> str:
    """Sucht nach relevanten Dokumenten basierend auf einer Anfrage."""
    docs = vectorstore.similarity_search(query, k=2)
    if docs:
        return "\n\n".join([doc.page_content for doc in docs])
    else:
        return "Keine relevanten Dokumente gefunden."


@tool
def umformuliere_anfrage(query: str, feedback: str) -> str:
    """
    Formuliert eine Suchanfrage um, um bessere Ergebnisse zu erzielen.
    Die Eingabe sollte die ursprüngliche Anfrage und Feedback enthalten.
    """
    prompt = ChatPromptTemplate.from_template(
        """Du bist ein Experte für Informationssuche. 
        Basierend auf der ursprünglichen Anfrage und dem Feedback, 
        formuliere eine verbesserte Suchanfrage.
        
        Ursprüngliche Anfrage: {query}
        Feedback: {feedback}
        
        Verbesserte Anfrage:"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "feedback": feedback})


@tool
def generiere_antwort(query: str, context: str) -> str:
    """
    Generiert eine Antwort basierend auf der Anfrage und dem gegebenen Kontext.
    """
    prompt = ChatPromptTemplate.from_template(
        """Du bist ein hilfreicher Assistent. 
        Beantworte die Frage basierend auf dem gegebenen Kontext.
        
        Kontext: {context}
        
        Frage: {query}
        
        Antwort:"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context})


# Agentic RAG initialisieren
tools = [suche_in_dokumenten, umformuliere_anfrage, generiere_antwort]

# Ihre Aufgabe: Vervollständigen Sie den Agenten und implementieren Sie eine Strategie zur Verbesserung der Suchergebnisse
# Tipp: Verwenden Sie das ReAct-Muster, um den Agenten bei seiner Entscheidungsfindung zu unterstützen

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Testen Sie den Agenten
result = agent.invoke(
    "Welche Städte in Deutschland haben mehr als 1 Million Einwohner?")
print(result)
```

### Erweiterungen:

- Implementieren Sie ein Tool zur Bewertung der Antwortqualität
- Fügen Sie ein Tool hinzu, das die Ergebnisse verschiedener Suchstrategien
  vergleicht
- Erweitern Sie die Vektordatenbank um zusätzliche Dokumente und testen Sie
  komplexere Anfragen

---

## Bonusübung: Implementieren eines Multi-Hop-RAG-Systems

### Aufgabe:

Entwickeln Sie ein RAG-System, das mehrere Hops (Suchschritte) nutzt, um
komplexe Fragen zu beantworten, die mehrere Fakten miteinander verknüpfen.

### Beispiel:

Bei der Frage "Welche Verbindung besteht zwischen dem Regisseur von 'Titanic'
und dem Hauptdarsteller von 'Inception'?" müsste das System:

1. Herausfinden, wer der Regisseur von "Titanic" ist (James Cameron)
2. Herausfinden, wer der Hauptdarsteller von "Inception" ist (Leonardo DiCaprio)
3. Die Verbindung zwischen beiden finden (z.B. gemeinsame Arbeit an "Titanic")

### Schritte:

1. Implementieren Sie ein System, das Fragen in Teilfragen zerlegt
2. Entwerfen Sie einen Prozess, der die Ergebnisse der Teilfragen zusammenführt
3. Implementieren Sie eine Strategie zur Verfolgung und Bewertung der "
   Hop-Pfade"

Diese Übung ist fortgeschritten und kombiniert Elemente aus allen vorgestellten
Architekturkonzepten.
