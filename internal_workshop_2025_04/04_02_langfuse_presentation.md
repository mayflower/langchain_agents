# LangFuse

## Monitoring und Tracing von LLM-Anwendungen

---

## Überblick

- Was ist LangFuse?
- Installation und Konfiguration
- Traces, Spans und Generations
- Integration mit LangChain
- Metriken und Evaluierung
- Anwendungsfälle und Best Practices

---

## Was ist LangFuse?

- Open-Source-Plattform für Observability von LLM-Anwendungen
- Ermöglicht Tracing, Monitoring und Evaluierung
- Unterstützt verschiedene LLMs und Frameworks
- Bietet eine UI für Analyse und Visualisierung
- Hilft bei der Optimierung von Prompts, Kosten und Leistung

![LangFuse UI](https://docs.langfuse.com/img/ui/trace-detail.jpg)

---

## Architektur und Komponenten

- **Traces**: Zeichnen vollständige Verarbeitungspfade auf
- **Spans**: Zeitsegmente innerhalb eines Trace
- **Generations**: Interaktionen mit LLMs (Prompt + Completion)
- **Feedback**: Qualitätsbewertungen für Antworten
- **Datasets**: Sammlungen von Beispielen für Evaluierung

---

## Installation und Konfiguration

```python
# Installation
pip install langfuse langchain-langfuse

# Initialisierung
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pfus-...",
    secret_key="sfus-...",
    host="https://cloud.langfuse.com"
)
```

Optionen für Hosting:
- LangFuse Cloud: Vollständig verwaltete Lösung
- Self-Hosted: Docker, Kubernetes, Vercel

---

## Einfaches Tracing

```python
# Trace erstellen
trace = langfuse.trace(
    name="einfacher_trace",
    tags=["demo", "workshop"],
    metadata={"benutzer_id": "12345"}
)

# Generation aufzeichnen
generation = trace.generation(
    name="erste_anfrage",
    model="gpt-4",
    prompt="Was ist LangFuse?",
    completion="LangFuse ist ein Monitoring-Tool..."
)

# Trace abschließen
trace.update(status="success")
```

---

## Spans: Operationen aufzeichnen

```python
# Span für eine größere Operation
with trace.span(name="datenverarbeitung") as span:
    # Operationen durchführen
    print("Verarbeite Daten...")
    time.sleep(1.5)
    
    # Optional: Metadaten aktualisieren
    span.update(metadata={
        "verarbeitete_datensätze": 42,
        "verarbeitungstyp": "text_chunking"
    })
```

Spans helfen, den Ablauf und die Dauer verschiedener Operationen zu verfolgen.

---

## Integration mit LangChain

```python
from langchain.callbacks import LangfuseCallbackHandler
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# LangFuse Callback-Handler
handler = LangfuseCallbackHandler()

# LLM mit Callback
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    callbacks=[handler]
)

# Chain ausführen
chain = LLMChain(llm=llm, prompt=template)
response = chain.invoke({"thema": "KI", "frage": "Was ist LangFuse?"})
```

---

## Komplexe Anwendungsfälle: RAG-Tracing

![RAG Tracing](https://docs.langfuse.com/img/ui/trace-detail-observability.jpg)

Struktur für RAG-Tracing:
1. Query-Verständnis
2. Dokumenten-Retrieval
3. Prompt-Erstellung
4. Antwortgenerierung

---

## Feedback und Evaluierung

```python
# Manuelles Feedback
feedback = langfuse.feedback(
    trace_id=trace.id,
    name="genauigkeit",
    value=0.9,  # 0-1 Skala
    comment="Die Antwort war sehr präzise."
)

# LLM-as-a-Judge Evaluierung
eval_feedback = langfuse.feedback(
    trace_id=trace.id,
    name="llm_bewertung",
    value=0.85,
    comment="Die Antwort war korrekt, aber unvollständig..."
)
```

---

## Wichtige Metriken in LangFuse

| Metrik | Beschreibung | Nutzen |
|--------|--------------|--------|
| Latenz | Zeit für Anfragen | Performance-Optimierung |
| Token-Nutzung | Verbrauch pro Anfrage | Kostenoptimierung |
| Erfolgsrate | Anteil erfolgreicher Anfragen | Zuverlässigkeitsüberwachung |
| Feedback-Werte | Qualitätsbewertungen | Qualitätsverbesserung |
| Trace-Vollständigkeit | Abdeckung des Tracings | Observability-Verbesserung |

---

## A/B-Testing mit LangFuse

```python
# Test-Variante A
trace_a = langfuse.trace(
    name="produktbeschreibung",
    tags=["ab_test", "variant_a"],
    metadata={"test_id": "test123", "variant": "A"}
)

# Test-Variante B
trace_b = langfuse.trace(
    name="produktbeschreibung",
    tags=["ab_test", "variant_b"],
    metadata={"test_id": "test123", "variant": "B"}
)
```

Vergleich in der UI:
- Kosten
- Latenz
- Feedback-Werte
- Token-Nutzung

---

## Dashboards und Analysen

![LangFuse Dashboard](https://docs.langfuse.com/img/ui/dashboard.jpg)

- Benutzerdefinierte Dashboards für verschiedene Anwendungsfälle
- Filterung nach Tags, Modellen, Status, etc.
- Trend-Analysen über Zeit
- Export von Daten für externe Analysen

---

## Best Practices für LangFuse

1. **Konsistente Benennung**
   - Verwenden Sie einheitliche Namen für Traces, Spans und Generations

2. **Sinnvolle Metadaten**
   - Fügen Sie relevante Kontext-Informationen hinzu

3. **Hierarchische Strukturierung**
   - Organisieren Sie Spans in einer logischen Hierarchie

4. **Feedback-Sammlung**
   - Sammeln Sie sowohl automatisiertes als auch manuelles Feedback

5. **Datenschutz beachten**
   - Vermeiden Sie das Speichern sensibler Daten

---

## Anwendungsfälle

- **Debugging**: Identifizieren problematischer Prompts oder Modelle
- **Kostenoptimierung**: Erkennen ineffizienter Token-Nutzung
- **Qualitätsverbesserung**: Systematische Evaluierung und Verbesserung von Antworten
- **Performance-Monitoring**: Überwachung von Latenz und Durchsatz
- **Produktions-Observability**: Kontinuierliche Überwachung von LLM-Anwendungen

---

## Zusammenfassung

- LangFuse bietet umfassendes Monitoring und Tracing für LLM-Anwendungen
- Einfache Integration in bestehende Anwendungen und Frameworks
- Ermöglicht datengestützte Optimierung von Prompts, Kosten und Qualität
- Unterstützt systematisches A/B-Testing und Evaluierung
- Open-Source mit Community-Support und aktiver Entwicklung

---

## Weitere Ressourcen

- [LangFuse Dokumentation](https://langfuse.com/docs)
- [GitHub Repository](https://github.com/langfuse/langfuse)
- [Discord Community](https://discord.gg/7NXusRtVa8)
- [LangChain Integration](https://python.langchain.com/docs/integrations/callbacks/langfuse)
