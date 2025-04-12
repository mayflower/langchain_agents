# Chaining - Verketten von Anfragen und Modellen

## Was ist Chaining?

- Technik zur Verkettung verschiedener Komponenten zu komplexen Workflows
- Ermöglicht die schrittweise Verarbeitung von Daten durch LLMs und andere
  Funktionen
- Fördert Modularität und Wiederverwendbarkeit von Code

---

## LangChain Expression Language (LCEL)

- Deklarative Syntax für die Komposition von LLM-Anwendungen
- Zentral: Der Pipe-Operator `|` zur Verkettung von Komponenten
- Macht Datenfluss klar lesbar und einfach zu verstehen

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein hilfreicher Assistent."),
    ("human", "{frage}")
])

# Verkettung mit dem Pipe-Operator
chain = prompt | llm() | StrOutputParser()

# Ausführen der Chain
ergebnis = chain.invoke({"frage": "Was ist KI?"})
```

---

## Vorteile des Chaining-Ansatzes

- **Modularität**: Komplexe Aufgaben in kleinere Komponenten zerlegen
- **Lesbarkeit**: Klare Darstellung des Datenflusses
- **Flexibilität**: Einfaches Austauschen einzelner Komponenten
- **Wiederverwendbarkeit**: Bausteine für unterschiedliche Anwendungen nutzen
- **Streaming-Support**: Native Unterstützung für Streaming-Antworten

---

## Von einfachen zu komplexen Chains

### Sequenzielle Verarbeitung

```python
# Text erst zusammenfassen, dann übersetzen
zusammenfassen_chain = zusammenfassung_prompt | llm() | StrOutputParser()
übersetzen_chain = übersetzung_prompt | llm() | StrOutputParser()

# Verkettung mit Zwischenschritten
komplexe_chain = (
        {"text": lambda x: x["original_text"]}
        | zusammenfassen_chain
        | (lambda x: {"text": x})
        | übersetzen_chain
)
```

---

## Parallele Verarbeitung mit RunnableMap

```python
from langchain.schema.runnable import RunnableMap

# Mehrere parallele Verarbeitungswege
parallel_chain = RunnableMap({
    "zusammenfassung": {"text": lambda x: x[
        "original_text"]} | zusammenfassen_chain,
    "übersetzung": {"text": lambda x: x["original_text"]} | übersetzen_chain
})

ergebnis = parallel_chain.invoke({"original_text": "Text..."})
print(f"Zusammenfassung: {ergebnis['zusammenfassung']}")
print(f"Übersetzung: {ergebnis['übersetzung']}")
```

---

## Bedingte Verarbeitung mit RunnableBranch

```python
from langchain.schema.runnable import RunnableBranch

# Verzweigung basierend auf Textklassifikation
bedingte_chain = RunnableBranch(
    (lambda x: "TECHNISCH" in klassifikation_chain.invoke({"text": x["text"]}),
     lambda x: technische_chain.invoke({"text": x["text"]})),

    (lambda x: "GESCHÄFTLICH" in klassifikation_chain.invoke(
        {"text": x["text"]}),
     lambda x: geschäftliche_chain.invoke({"text": x["text"]})),

    # Fallback für alle anderen Fälle
    lambda x: kreative_chain.invoke({"text": x["text"]})
)
```

---

## Streaming mit LCEL

- Ideal für lange Antworten - zeigt Ergebnisse inkrementell
- Verbessert die Nutzererfahrung durch reduzierte Wartezeiten
- Einfache Integration in Web-Anwendungen

```python
async def stream_text():
    async for chunk in chain.astream({"thema": "KI-Entwicklung"}):
        print(chunk, end="", flush=True)
        # In einer Webanwendung könnte jeder Chunk an den Client gesendet werden
```

---

## Beispiel eines praktischen Workflows

Text-Analyse-Pipeline mit mehreren Schritten:

1. Zusammenfassung erstellen
2. Detaillierte Analyse mit strukturierter Ausgabe
3. Empfehlungen basierend auf Analyse generieren

```python
# Komplette Pipeline
def text_analyse_pipeline(text):
    # Schritt 1: Zusammenfassung
    zusammenfassung = zusammenfassung_chain.invoke({"text": text})

    # Schritt 2: Detaillierte Analyse
    analyse = analyse_chain.invoke({"text": text})

    # Schritt 3: Empfehlungen
    empfehlungen = empfehlungen_chain.invoke({
        "zusammenfassung": zusammenfassung,
        "analyse": analyse.json()
    })

    return {...}  # Ergebnisse zusammenführen
```

---

## Best Practices für effektives Chaining

1. **Modularer Aufbau**: Kleine, spezialisierte Komponenten erstellen
2. **Fehlerbehandlung**: Robuste Fehlerbehandlung für jede Komponente
3. **Typsicherheit**: Klare Ein- und Ausgabetypen definieren
4. **Monitoring**: LangSmith oder Langfuse für Tracing einsetzen
5. **Caching**: Rechenintensive Operationen cachen
6. **Testen**: Einzelne Komponenten isoliert testen

---

## Zusammenfassung: Chaining in LangChain

- **Strukturierter Ansatz** für komplexe LLM-Anwendungen
- **LCEL und Pipe-Operator** für intuitive Verkettung
- **Flexible Muster**: Sequenziell, parallel, bedingt
- **Streaming-Unterstützung** für bessere Nutzererfahrung
- **Grundlage** für fortgeschrittene Architekturkonzepte

---

## Fragen?
