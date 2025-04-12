# Chaining in LangChain: Verketten von Modellen und Funktionen

## Was ist Chaining?

Chaining ist ein fundamentales Konzept in LangChain, das es ermöglicht,
verschiedene Komponenten (LLMs, Tools, Parser, etc.) miteinander zu verketten,
um komplexe Workflows zu erstellen. Durch Chaining können wir:

1. **Modularität erreichen**: Aufteilung komplexer Aufgaben in kleinere,
   wiederverwendbare Komponenten
2. **Flexibilität gewinnen**: Einfaches Austauschen oder Anpassen einzelner
   Komponenten
3. **Pipeline-Verarbeitung**: Sequenzielle Verarbeitung von Daten durch mehrere
   Verarbeitungsschritte

## LangChain Expression Language (LCEL)

Die LangChain Expression Language (LCEL) ist eine deklarative Methode zum
Erstellen von Chains mit dem Pipe-Operator `|`.

### Grundlegende Syntax

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from helpers import llm

# Einfache Komponenten definieren
prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein hilfreicher Assistent."),
    ("human", "{frage}")
])

# Mit dem Pipe-Operator (|) verbinden
chain = prompt | llm() | StrOutputParser()

# Chain ausführen
ergebnis = chain.invoke({"frage": "Was ist Machine Learning?"})
print(ergebnis)
```

### Vorteile von LCEL

- **Lesbarkeit**: Der Pipe-Operator macht den Datenfluss klar erkennbar
- **Komposition**: Einfaches Zusammenfügen von Komponenten
- **Typsicherheit**: Bessere Fehlerbehandlung durch klare Schnittstellen

## Von einfachen zu komplexen Chains

### 1. Einfache sequenzielle Chain

```python
# Einfache sequenzielle Verarbeitung
zusammenfassung_prompt = ChatPromptTemplate.from_messages([
    ("system", "Fasse den folgenden Text in 3 Sätzen zusammen."),
    ("human", "{text}")
])

übersetzung_prompt = ChatPromptTemplate.from_messages([
    ("system", "Übersetze den folgenden Text ins Deutsche."),
    ("human", "{text}")
])

# Text erst zusammenfassen, dann übersetzen
zusammenfassen_chain = zusammenfassung_prompt | llm() | StrOutputParser()
übersetzen_chain = übersetzung_prompt | llm() | StrOutputParser()

# Chains verknüpfen mit Zwischenspeicherung des Ergebnisses
komplexe_chain = (
        {"text": lambda x: x["original_text"]}
        | zusammenfassen_chain
        | (lambda x: {"text": x})
        | übersetzen_chain
)

ergebnis = komplexe_chain.invoke({
                                     "original_text": "Large Language Models (LLMs) are a type of machine learning model that can process and generate natural language. They are trained on vast amounts of text data using neural networks with many parameters. LLMs have shown remarkable abilities in various tasks such as translation, summarization, and creative writing, often exhibiting emergent capabilities beyond what they were explicitly trained for."})
```

### 2. Verzweigte Chains

```python
# Verzweigungen mit RunnableMap erstellen
from langchain.schema.runnable import RunnableMap

# Zwei parallele Verarbeitungswege
parallel_chain = RunnableMap({
    "zusammenfassung": {"text": lambda x: x[
        "original_text"]} | zusammenfassen_chain,
    "übersetzung": {"text": lambda x: x["original_text"]} | übersetzen_chain
})

ergebnis = parallel_chain.invoke({
                                     "original_text": "LangChain is a framework designed to simplify the creation of applications using large language models."})
print(f"Zusammenfassung: {ergebnis['zusammenfassung']}")
print(f"Übersetzung: {ergebnis['übersetzung']}")
```

### 3. Streaming von Ergebnissen

Ein wichtiger Vorteil von LCEL ist die integrierte Unterstützung für Streaming,
was besonders bei langsamen LLM-Antworten hilfreich ist:

```python
async def stream_ergebnis():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Geschichtenerzähler."),
        ("human", "Erzähle eine kurze Geschichte über {thema}")
    ])

    chain = prompt | llm() | StrOutputParser()

    async for chunk in chain.astream(
            {"thema": "einen Roboter, der Gefühle entwickelt"}):
        print(chunk, end="", flush=True)
        # In einer Webanwendung könnten wir hier Chunks an den Client senden
```

## Best Practices für effektives Chaining

1. **Modularität**: Erstellen Sie kleine, wiederverwendbare Komponenten
2. **Fehlerbehandlung**: Implementieren Sie robuste Fehlerbehandlung für jede
   Komponente
3. **Typsicherheit**: Verwenden Sie klare Ein- und Ausgabetypen
4. **Tracing**: Nutzen Sie LangSmith oder ähnliche Tools zur Nachverfolgung der
   Chain-Ausführung
5. **Caching**: Implementieren Sie Caching für rechenintensive Operationen

## Beispiel eines praktischen Workflows

Hier ein Beispiel einer Chain, die einen Text analysiert, die Stimmung bewertet
und eine Zusammenfassung erstellt:

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Ausgabemodell für die Stimmungsanalyse
class StimmungsAnalyse(BaseModel):
    stimmung: str = Field(
        description="Die allgemeine Stimmung des Textes (positiv, neutral, negativ)")
    wichtige_punkte: List[str] = Field(
        description="Die 3 wichtigsten Punkte aus dem Text")


# Parser für strukturierte Ausgabe
parser = PydanticOutputParser(pydantic_object=StimmungsAnalyse)

# Prompt für die Stimmungsanalyse
analyse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Analysiere den folgenden Text und extrahiere die Stimmung sowie die wichtigsten Punkte.\n\n{format_instructions}"),
    ("human", "{text}")
])

# Prompt für die Zusammenfassung
zusammenfassung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Basierend auf der folgenden Analyse, erstelle eine kurze Zusammenfassung des Originaltextes."),
    (
    "human", "Analyse: {analyse}\n\nErstelle eine Zusammenfassung in 2 Sätzen.")
])

# Chain erstellen
analyse_chain = analyse_prompt.partial(
    format_instructions=parser.get_format_instructions()) | llm() | parser
zusammenfassung_chain = zusammenfassung_prompt | llm() | StrOutputParser()

# Komplette Analyse-Pipeline
analyse_workflow = (
        {"text": lambda x: x["text"]}
        | analyse_chain
        | (lambda x: {"analyse": x.json()})
        | zusammenfassung_chain
)

# Ausführen
text = """
Die Einführung des neuen Produkts übertraf alle Erwartungen. Die Verkaufszahlen im ersten Quartal waren um 45% höher als prognostiziert, und das Kundenfeedback war überwiegend positiv. Allerdings gab es einige Beschwerden über die Lieferzeit und vereinzelte Qualitätsprobleme, die behoben werden müssen.
"""

ergebnis = analyse_workflow.invoke({"text": text})
print(ergebnis)
```

## Zusammenfassung

Chaining in LangChain:

- Erlaubt die Verkettung von LLMs, Tools und Funktionen
- Nutzt den Pipe-Operator für lesbare, modulare Kompositionen
- Ermöglicht die Entwicklung komplexer, flexibler KI-Anwendungen
- Unterstützt Streaming, parallele Verarbeitung und verzweigte Flows

Diese Konzepte bilden die Grundlage für fortgeschrittenere Architekturen wie
agentenbasierte Systeme und komplexe RAG-Implementierungen.
