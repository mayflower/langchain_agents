# Prompt-Management - Handout

## Überblick

Effektives Prompt-Management ist entscheidend für erfolgreiche LLM-Anwendungen,
besonders in Teams und Produktionsumgebungen. In diesem Modul lernen Sie, wie
Sie Prompts organisieren, versionieren und in Ihrer Entwicklungsumgebung
effizient verwalten können.

## Grundlagen des Prompt-Managements

### Warum ist Prompt-Management wichtig?

- **Konsistenz**: Stellt sicher, dass alle Teammitglieder die gleichen,
  optimierten Prompts verwenden
- **Versionierung**: Ermöglicht die Nachverfolgung von Änderungen und das
  Zurückkehren zu früheren Versionen
- **Wiederverwendbarkeit**: Fördert die Nutzung bewährter Prompts in
  verschiedenen Anwendungsfällen
- **Qualitätssicherung**: Erleichtert das Testen und die Qualitätskontrolle von
  Prompts
- **Dokumentation**: Hilft dabei, Entscheidungen und Optimierungen zu
  dokumentieren

### Herausforderungen ohne Prompt-Management

- Inkonsistente Ergebnisse zwischen verschiedenen Entwicklern oder
  Anwendungsteilen
- "Prompt-Drift" - allmähliche, undokumentierte Änderungen über Zeit
- Schwierigkeiten bei der Fehlerbehebung, wenn Prompt-Änderungen nicht
  nachverfolgt werden
- Verlorenes Wissen über erfolgreiche Prompt-Strategien
- Ineffiziente Wiederholung ähnlicher Prompt-Entwicklungsarbeit

## Organisation und Versionierung von Prompts

### Strategien für die Prompt-Organisation

1. **Zentrale Prompt-Bibliothek**:
    - Erstellen einer zentralen Sammlung aller produktiven Prompts
    - Kategorisierung nach Anwendungsfällen oder Funktionsbereichen

2. **Template-System**:
   ```python
   from langchain.prompts import ChatPromptTemplate
   
   # Beispiel für ein Prompt-Template
   customer_service_template = ChatPromptTemplate.from_messages([
       ("system", "Du bist ein freundlicher Kundenservice-Mitarbeiter der {firma}. Du hilfst Kunden mit Problemen bei {produkt}."),
       ("human", "{kundenanfrage}")
   ])
   
   # Verwendung des Templates
   prompt = customer_service_template.format(
       firma="TechCorp",
       produkt="SmartHome-System",
       kundenanfrage="Mein Gerät verbindet sich nicht mit dem WLAN."
   )
   ```

3. **Prompt-Versionierung**:
    - Eindeutige Identifikation von Prompts (z.B. durch Namen und Versionen)
    - Dokumentation von Änderungen und Verbesserungen
    - Integration mit Versionskontrollsystemen (Git)

### Best Practices für Prompt-Verwaltung

- **Naming-Konventionen**: Klare, beschreibende Namen für Prompts
- **Kommentierung**: Erläuterung des Zwecks und der Funktionsweise eines Prompts
- **Metadaten**: Zusätzliche Informationen wie Autor, Erstellungsdatum,
  Anwendungsbereich
- **Optimierungshistorie**: Aufzeichnung von Verbesserungen und deren
  Auswirkungen

## LangChain Hub für Prompt-Sharing

LangChain Hub ist eine Plattform zum Speichern, Verwalten und Teilen von Prompts
und anderen LangChain-Komponenten.

### Funktionen des LangChain Hub

- Zentrales Repository für Prompts und andere LangChain-Objekte
- Versionskontrolle für LangChain-Komponenten
- Einfache Integration in LangChain-Anwendungen
- Teambasierte Zusammenarbeit an Prompts

### Arbeiten mit LangChain Hub

#### Prompts veröffentlichen

```python
from langchain.prompts import ChatPromptTemplate
from langchain import hub

# Prompt erstellen
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Experte für Sentiment-Analyse."),
    ("human",
     "Analysiere den folgenden Text und gib das Sentiment als 'positiv', 'neutral' oder 'negativ' an: {text}")
])

# Prompt im Hub veröffentlichen
hub.push("mein-username/sentiment-analysis", sentiment_prompt)
```

#### Prompts aus dem Hub laden

```python
from langchain import hub

# Prompt aus dem Hub laden
sentiment_prompt = hub.pull("mein-username/sentiment-analysis")

# Alternativ kann auch ein Community-Prompt geladen werden
customer_sentiment = hub.pull("borislove/customer-sentiment-analysis")

# Prompt verwenden
formatted_prompt = customer_sentiment.format(
    client_letter="Ich bin sehr zufrieden mit dem Produkt...",
    format_instructions="Analysiere das Sentiment auf einer Skala von 1-5..."
)
```

### Vorteile der Hub-Nutzung

- **Zentralisierung**: Ein zentraler Ort für alle Prompts
- **Kollaboration**: Einfaches Teilen und gemeinsames Bearbeiten von Prompts
- **Community**: Zugriff auf bewährte Prompts der Community
- **Versionierung**: Automatische Versionierung beim Aktualisieren von Prompts

## Integration mit Langfuse für Monitoring und Tracking

[Langfuse](https://langfuse.com/) ist ein Tool für Observability und Monitoring
von LLM-Anwendungen, das sich nahtlos in LangChain integrieren lässt.

### Einrichtung von Langfuse

```python
import os
from langfuse import Langfuse
from langfuse.client import StatelessTracer

# Langfuse initialisieren
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
```

### Tracing von Prompts und Antworten

```python
# Tracing einer Prompt-Ausführung
tracer = StatelessTracer(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Trace starten
trace = tracer.trace(name="sentiment_analysis_trace")

# Prompt-Ausführung tracen
generation = trace.generation(
    name="sentiment_analysis",
    model="gpt-4",
    prompt="Analysiere das Sentiment: 'Ich bin sehr zufrieden mit dem Service.'",
    completion="Das Sentiment ist positiv, da der Kunde explizit seine Zufriedenheit ausdrückt."
)

# Metadaten hinzufügen
generation.update(
    metadata={
        "prompt_version": "1.2",
        "prompt_template_id": "sentiment_analysis_v1",
        "application": "customer_feedback_system"
    }
)
```

### Integration mit LangChain

```python
from langchain.callbacks import LangfuseCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Langfuse Callback Handler erstellen
langfuse_handler = LangfuseCallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

# LLM mit Langfuse Callback
llm = OpenAI(temperature=0.7, callbacks=[langfuse_handler])

# Chain erstellen
chain = LLMChain(
    llm=llm,
    prompt=sentiment_prompt,
    callbacks=[langfuse_handler]
)

# Chain ausführen (wird automatisch getrackt)
result = chain.run(text="Der Kundenservice war fantastisch!")
```

### Vorteile des Prompt-Trackings mit Langfuse

- **Prompt-Performance**: Verfolgen, wie gut verschiedene Prompt-Versionen
  funktionieren
- **Kostenanalyse**: Überwachen der Kosten pro Prompt-Aufruf
- **Latenz-Tracking**: Messen der Antwortzeiten
- **Fehlerdiagnose**: Identifizieren von Problemen mit bestimmten Prompts
- **Nutzungsmuster**: Verstehen, welche Prompts am häufigsten verwendet werden

## Best Practices für Prompt-Management in größeren Teams

1. **Klare Ownership**: Definieren Sie Verantwortliche für bestimmte
   Prompt-Kategorien oder -Bereiche

2. **Review-Prozess**: Implementieren Sie einen Peer-Review-Prozess für
   Prompt-Änderungen
   ```
   1. Prompt-Entwicklung in einer Entwicklungsumgebung
   2. Review durch Team-Mitglieder
   3. A/B-Testing von Prompt-Varianten
   4. Freigabe für Produktion
   ```

3. **Dokumentation**:
    - Zweck des Prompts
    - Erwartetes Verhalten
    - Edge Cases und Einschränkungen
    - Änderungshistorie und Gründe für Änderungen

4. **Checkpoints mit LangSmith**:
   ```python
   from langchain.smith import RunEvalConfig, run_on_dataset
   
   # Checkpoint-Evaluation
   eval_config = RunEvalConfig(
       evaluators=[
           "qa",
           "criteria:relevance",
           "criteria:coherence"
       ]
   )
   
   # Ausführen der Evaluation
   results = run_on_dataset(
       dataset_name="prompt_validation_set",
       llm_or_chain=chain,
       evaluation=eval_config
   )
   ```

5. **Prompt-Governance**:
    - Richtlinien für Sicherheit, Fairness und Ethik
    - Regelmäßige Audits und Überprüfungen
    - Vorgaben für sensible Anwendungsbereiche

## Praktische Tipps für die Implementierung

1. **Start klein**: Beginnen Sie mit einem einfachen System und erweitern Sie es
   nach Bedarf
2. **Automatisierung**: Automatisieren Sie Routineaspekte des Prompt-Managements
3. **Feedback-Schleife**: Etablieren Sie einen Prozess zur kontinuierlichen
   Verbesserung
4. **Schulung**: Stellen Sie sicher, dass alle Teammitglieder mit dem
   Prompt-Management-System vertraut sind
5. **Ausfallsicherheit**: Planen Sie für den Fall von API-Ausfällen oder
   Versionsproblemen

## Zusammenfassung

Effektives Prompt-Management ist ein entscheidender Faktor für erfolgreiche
LLM-Anwendungen, besonders in Teams oder Produktionsumgebungen. Durch die
Nutzung von Tools wie LangChain Hub und Langfuse können Sie Prompts
organisieren, versionieren und deren Performance überwachen. Die Implementierung
klarer Prozesse und Best Practices hilft dabei, die Qualität und Konsistenz
Ihrer Prompts über die gesamte Anwendung hinweg sicherzustellen.

## Übungen und Weiterführende Ressourcen

### Übung 1: Prompt-Versionierung

Erstellen Sie eine Prompt-Template-Bibliothek mit mindestens zwei Versionen
eines Prompts und dokumentieren Sie die Unterschiede.

### Übung 2: LangChain Hub

Veröffentlichen Sie einen eigenen Prompt im LangChain Hub und stellen Sie
sicher, dass Sie ihn in einer Anwendung laden können.

### Übung 3: Prompt-Tracking

Integrieren Sie Langfuse in eine einfache LangChain-Anwendung und analysieren
Sie die Leistungsdaten für verschiedene Prompt-Versionen.

### Weiterführende Ressourcen

- [LangChain Hub Dokumentation](https://smith.langchain.com/hub)
- [Langfuse Dokumentation](https://langfuse.com/docs)
- [LangSmith für Prompt-Evaluation](https://smith.langchain.com/)
- [Artikel: "Effective Prompt Engineering Practices"](https://www.promptingguide.ai/)
