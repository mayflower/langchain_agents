# LangFuse - Übungen

## Übung 1: Einfaches Tracing mit LangFuse

Ziel dieser Übung ist es, LangFuse für das Tracing einer einfachen Konversation mit einem LLM zu verwenden.

### Aufgaben:

1. Initialisieren Sie den LangFuse-Client mit Ihren API-Schlüsseln
2. Erstellen Sie einen Trace für eine Konversation
3. Führen Sie mindestens 3 Nachrichtenwechsel durch und zeichnen Sie jeden als Generation auf
4. Fügen Sie relevante Metadaten und Tags hinzu
5. Beenden Sie den Trace mit einem Status

### Code-Gerüst:

```python
from langfuse import Langfuse
from openai import OpenAI
import os
import time

# TODO: Initialisieren Sie den LangFuse-Client

# TODO: Erstellen Sie einen Trace mit geeignetem Namen und Tags

# Initialisieren Sie den OpenAI-Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Konversationshistorie
conversation_history = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."}
]

# TODO: Implementieren Sie eine Funktion für den Nachrichtenwechsel, die jede 
# Generation in LangFuse aufzeichnet

# TODO: Führen Sie mindestens 3 Nachrichtenwechsel durch
messages = [
    "Hallo! Ich möchte etwas über Vektordatenbanken lernen.",
    "Was sind die wichtigsten Vorteile von Vektordatenbanken für RAG-Anwendungen?",
    "Kannst du mir Beispiele für populäre Vektordatenbanken nennen?"
]

# TODO: Beenden Sie den Trace mit einem erfolgreichen Status
```

## Übung 2: LangChain-Integration und Evaluierung

In dieser Übung werden Sie LangFuse mit LangChain integrieren und automatisierte Evaluierungen implementieren.

### Aufgaben:

1. Erstellen Sie eine einfache LangChain-Chain mit dem LangFuse-Callback-Handler
2. Führen Sie mehrere Anfragen mit unterschiedlichen Eingaben aus
3. Implementieren Sie eine automatisierte Evaluierung mit dem LLM-as-a-Judge-Ansatz
4. Fügen Sie das Evaluierungsergebnis als Feedback zum Trace hinzu
5. Überprüfen Sie die Ergebnisse in der LangFuse-UI

### Code-Gerüst:

```python
from langfuse import Langfuse
from langchain.callbacks import LangfuseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# LangFuse initialisieren
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# TODO: Erstellen Sie den LangFuse-Callback-Handler

# TODO: Initialisieren Sie das LLM mit dem Callback-Handler

# TODO: Erstellen Sie ein Prompt-Template für eine Zusammenfassungsaufgabe

# TODO: Erstellen Sie eine Chain mit dem LLM und dem Prompt-Template

# Testdaten
test_texts = [
    "Vektordatenbanken sind spezialisierte Datenbanksysteme, die für die effiziente Speicherung und Abfrage von hochdimensionalen Vektoren optimiert sind...",
    "Large Language Models (LLMs) wie GPT-4 und Claude haben die Art und Weise, wie wir mit Technologie interagieren, revolutioniert...",
    "RAG (Retrieval Augmented Generation) ist ein Ansatz, der die Stärken von Retrievalsystemen und generativen Modellen kombiniert..."
]

# TODO: Führen Sie für jeden Testtext die Chain aus und speichern Sie die Ergebnisse

# TODO: Implementieren Sie eine Funktion zur automatisierten Evaluierung der Zusammenfassungen

# TODO: Fügen Sie die Evaluierungsergebnisse als Feedback zu den entsprechenden Traces hinzu
```

## Übung 3: Komplexes RAG-Tracing

In dieser Übung werden Sie ein detailliertes Tracing für einen RAG-Workflow implementieren.

### Aufgaben:

1. Erstellen Sie eine einfache RAG-Pipeline mit folgenden Komponenten:
   - Query-Analyse
   - Dokumenten-Retrieval (simuliert oder mit einer echten Vektordatenbank)
   - Prompt-Erstellung
   - Antwortgenerierung
2. Implementieren Sie ein detailliertes Tracing mit hierarchischen Spans für jeden Schritt
3. Zeichnen Sie relevante Metriken und Metadaten für jeden Schritt auf
4. Evaluieren Sie die Qualität des RAG-Ergebnisses

### Code-Gerüst:

```python
from langfuse.client import StatelessTracer
from openai import OpenAI
import time
import os

# Initialisieren des Tracers
tracer = StatelessTracer(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# OpenAI-Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# TODO: Erstellen Sie einen Trace für den RAG-Workflow

# Beispielanfrage
query = "Was sind die Hauptunterschiede zwischen Chroma und Qdrant?"

# TODO: Implementieren Sie den Query-Analyse-Schritt mit einem Span

# Simulierte Dokumente (in einer realen Anwendung würden Sie hier eine Vektordatenbank verwenden)
documents = [
    "Chroma ist eine eingebettete Vektordatenbank, die für Prototyping und kleine Projekte optimiert ist...",
    "Qdrant ist eine vektorbasierte Suchengine, die für Produktion und Hochverfügbarkeit ausgelegt ist...",
    "Vektordatenbanken speichern Embeddings als numerische Vektoren und ermöglichen Ähnlichkeitssuche..."
]

# TODO: Implementieren Sie den Retrieval-Schritt mit einem Span

# TODO: Implementieren Sie den Prompt-Erstellungsschritt mit einem Span

# TODO: Implementieren Sie den Antwortgenerierungsschritt mit einem Span

# TODO: Implementieren Sie eine Evaluierung der Antwort

# TODO: Beenden Sie den Trace
```

## Übung 4: A/B-Testing von Prompt-Strategien

In dieser Übung werden Sie LangFuse für das A/B-Testing verschiedener Prompt-Strategien verwenden.

### Aufgaben:

1. Definieren Sie mindestens zwei verschiedene Prompt-Strategien für die gleiche Aufgabe
2. Implementieren Sie ein Tracing-System für A/B-Tests mit LangFuse
3. Führen Sie mehrere Tests mit verschiedenen Eingaben durch
4. Sammeln Sie Metriken wie Token-Verbrauch, Latenz und Qualität
5. Analysieren Sie die Ergebnisse in der LangFuse-UI

### Code-Gerüst:

```python
from langfuse import Langfuse
from openai import OpenAI
import time
import os
import random

# Initialisieren der Clients
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Test-ID generieren
test_id = f"prompt_test_{int(time.time())}"

# Verschiedene Prompt-Strategien definieren
prompt_strategies = {
    "standard": "Beantworte die folgende Frage klar und präzise: {query}",
    "detailliert": "Gib eine ausführliche Erklärung zur folgenden Frage. Berücksichtige verschiedene Aspekte und gib konkrete Beispiele: {query}",
    "schrittweise": "Beantworte die folgende Frage in klar definierten Schritten. Nummeriere jeden Schritt: {query}"
}

# Testfragen
test_queries = [
    "Wie funktionieren Vektordatenbanken?",
    "Was sind die Vorteile von RAG gegenüber reinen LLM-Anwendungen?",
    "Erkläre den HNSW-Algorithmus für Vektorsuche."
]

# TODO: Implementieren Sie eine Funktion zum Durchführen eines A/B-Tests für eine einzelne Frage

# TODO: Führen Sie Tests für alle Fragen und Prompt-Strategien durch

# TODO: Fügen Sie eine Funktion zur Evaluierung der Antwortqualität hinzu
```

## Übung 5: Produktions-Monitoring-Setup

In dieser fortgeschrittenen Übung werden Sie ein robustes Monitoring-Setup für eine Produktionsumgebung implementieren.

### Aufgaben:

1. Erstellen Sie eine Wrapper-Klasse für LangFuse, die eine fehlertolerante Tracing-Lösung bietet
2. Implementieren Sie Datenschutz-Features wie PII-Maskierung
3. Entwickeln Sie einen asynchronen Logging-Mechanismus
4. Definieren Sie sinnvolle Metriken und Alerts

### Code-Gerüst:

```python
from langfuse import Langfuse
from langfuse.client import StatelessTracer
import asyncio
import logging
import os
import re
from typing import Dict, List, Any, Optional, Union
from functools import wraps

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langfuse_monitoring")

class ProductionLangFuseTracer:
    def __init__(self, public_key: str, secret_key: str, host: str):
        """Initialisieren des Production Tracers mit Fehlerbehandlung."""
        try:
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self.tracer = StatelessTracer(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self.enabled = True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren von LangFuse: {str(e)}")
            self.enabled = False
    
    # TODO: Implementieren Sie eine PII-Maskierungsfunktion
    
    # TODO: Implementieren Sie eine fehlertolerante trace()-Methode
    
    # TODO: Implementieren Sie eine asynchrone Feedback-Sammlung
    
    # TODO: Erstellen Sie einen Decorator für einfaches Funktions-Tracing
    
    # TODO: Implementieren Sie eine Methode zur Batch-Verarbeitung von Traces

# Beispielnutzung:
# tracer = ProductionLangFuseTracer(
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
#     host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
# )
```

## Bonus-Übung: Integration mit einem eigenen Projekt

Wenn Sie an einem eigenen LLM-Projekt arbeiten, versuchen Sie, LangFuse darin zu integrieren. Achten Sie dabei auf folgende Aspekte:

1. Welche Operationen sollten getrackt werden?
2. Welche Metriken sind für Ihr Projekt am wichtigsten?
3. Wie können Sie die Qualität Ihrer LLM-Antworten automatisiert evaluieren?
4. Welche Optimierungsmöglichkeiten können Sie auf Basis der LangFuse-Daten identifizieren?

Dokumentieren Sie Ihren Implementierungsansatz und die gewonnenen Erkenntnisse.
