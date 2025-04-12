# LangFuse - Handout

## Was ist LangFuse?

LangFuse ist eine Open-Source-Plattform für das Monitoring, Tracing und die Evaluierung von LLM-Anwendungen. Es ermöglicht Entwicklern, die Leistung, Qualität und Kosten ihrer KI-Anwendungen zu überwachen und zu optimieren.

## Grundlegende Konzepte

### Traces

Ein Trace repräsentiert einen vollständigen Verarbeitungspfad, z.B. eine Benutzeranfrage und alle damit verbundenen Operationen.

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pfus-...",
    secret_key="sfus-...",
    host="https://cloud.langfuse.com"
)

trace = langfuse.trace(
    name="benutzeranfrage",
    tags=["produktion", "kundensupport"],
    metadata={"benutzer_id": "user123"}
)
```

### Generations

Eine Generation repräsentiert eine Interaktion mit einem LLM, einschließlich Prompt und Completion.

```python
generation = trace.generation(
    name="erste_antwort",
    model="gpt-4",
    prompt="Erstelle eine Zusammenfassung des folgenden Textes: ...",
    completion="Der Text beschreibt...",
    usage={
        "prompt_tokens": 150,
        "completion_tokens": 80,
        "total_tokens": 230
    }
)
```

### Spans

Ein Span repräsentiert eine zeitlich begrenzte Operation innerhalb eines Trace.

```python
# Einen Span für eine Operation erstellen
with trace.span(name="datenbankabfrage") as span:
    # Operation durchführen
    result = database.query("SELECT * FROM users")
    
    # Span mit Metadaten aktualisieren
    span.update(
        metadata={
            "ergebnisse": len(result),
            "abfrage_typ": "benutzersuche"
        }
    )
```

## Integration mit LangChain

LangFuse kann nahtlos in LangChain integriert werden, um Chains, Agenten und andere Komponenten zu tracen.

```python
from langchain.callbacks import LangfuseCallbackHandler
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# LangFuse Callback-Handler erstellen
handler = LangfuseCallbackHandler(
    public_key="pfus-...",
    secret_key="sfus-...",
    host="https://cloud.langfuse.com"
)

# LLM mit Callback-Handler initialisieren
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    callbacks=[handler]
)

# Prompt-Template erstellen
template = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Experte für {thema}."),
    ("human", "{frage}")
])

# Chain erstellen und ausführen
chain = LLMChain(llm=llm, prompt=template)
response = chain.invoke({
    "thema": "künstliche Intelligenz",
    "frage": "Was sind die neuesten Entwicklungen im Bereich der LLMs?"
})
```

## Feedback und Evaluierung

LangFuse ermöglicht das Sammeln von Feedback und die Evaluierung von LLM-Antworten.

```python
# Manuelles Feedback hinzufügen
feedback = langfuse.feedback(
    trace_id=trace.id,
    name="nuetzlichkeit",
    value=0.75,  # Wert zwischen 0 und 1
    comment="Die Antwort war hilfreich, aber nicht vollständig."
)

# Automatisierte Evaluierung mit LLM-as-a-Judge
eval_feedback = langfuse.feedback(
    trace_id=trace.id,
    name="korrektheit",
    value=0.9,
    comment="Die Antwort enthält alle wichtigen Informationen und ist faktisch korrekt.",
    metadata={
        "evaluator": "gpt-4",
        "kriterien": ["faktische_korrektheit", "vollstaendigkeit"]
    }
)
```

## A/B-Testing

LangFuse eignet sich hervorragend für A/B-Testing verschiedener Prompt-Strategien oder Modelle.

```python
# Test-ID für den Vergleich
test_id = "prompt_test_123"

# Variante A
trace_a = langfuse.trace(
    name="produktbeschreibung",
    tags=["ab_test", "variant_a"],
    metadata={"test_id": test_id, "variant": "A"}
)

# Variante B
trace_b = langfuse.trace(
    name="produktbeschreibung",
    tags=["ab_test", "variant_b"],
    metadata={"test_id": test_id, "variant": "B"}
)

# Nach dem Test können die Varianten in der LangFuse-UI verglichen werden
```

## Metriken und Dashboards

LangFuse bietet verschiedene Metriken zur Überwachung von LLM-Anwendungen:

- **Latenz**: Reaktionszeit der LLM-Anfragen
- **Token-Nutzung**: Verbrauch an Tokens pro Anfrage
- **Kosten**: Geschätzte Kosten basierend auf der Token-Nutzung
- **Erfolgsrate**: Anteil erfolgreicher Anfragen
- **Qualitätsmetriken**: Aggregierte Feedback-Werte

## Best Practices

1. **Strukturierte Traces**
   - Verwenden Sie konsistente Namenskonventionen
   - Organisieren Sie Spans in einer logischen Hierarchie
   - Nutzen Sie Tags und Metadaten für einfache Filterung

2. **Umfassendes Tracing**
   - Tracen Sie alle wichtigen Operationen, nicht nur LLM-Aufrufe
   - Schließen Sie auch Vorverarbeitungs- und Nachbearbeitungsschritte ein
   - Zeichnen Sie relevante Metriken und Metadaten auf

3. **Feedback-Sammlung**
   - Integrieren Sie automatisierte Evaluierungsmethoden
   - Sammeln Sie Benutzerfeedback, wenn möglich
   - Definieren Sie klare Kriterien für die Qualitätsbewertung

4. **Datenschutz**
   - Vermeiden Sie das Speichern sensibler Daten in Prompts/Completions
   - Implementieren Sie PII-Maskierung bei Bedarf
   - Setzen Sie Data Retention Policies um

## Anwendungsfälle

- **Debugging**: Identifizieren problematischer Prompts oder Modelle
- **Kostenoptimierung**: Erkennen ineffizienter Token-Nutzung
- **Qualitätsverbesserung**: Systematische Evaluierung und Verbesserung
- **Performance-Monitoring**: Überwachung von Latenz und Durchsatz
- **Dokumentation**: Speichern von Interaktionen für Audit-Zwecke

## Stateless Tracing für komplexe Anwendungen

Für komplexere Anwendungen bietet LangFuse einen Stateless Tracer:

```python
from langfuse.client import StatelessTracer

# Stateless Tracer initialisieren
tracer = StatelessTracer(
    public_key="pfus-...",
    secret_key="sfus-...",
    host="https://cloud.langfuse.com"
)

# Trace und Spans erstellen
trace = tracer.trace(name="komplexe_anwendung")
with trace.span(name="hauptprozess") as main_span:
    # Untergeordnete Spans
    with main_span.span(name="teilprozess_1") as sub_span:
        # Operationen...
        pass
```

## Weitere Ressourcen

- [LangFuse Dokumentation](https://langfuse.com/docs)
- [GitHub Repository](https://github.com/langfuse/langfuse)
- [Discord Community](https://discord.gg/7NXusRtVa8)
- [LangChain Integration](https://python.langchain.com/docs/integrations/callbacks/langfuse)
