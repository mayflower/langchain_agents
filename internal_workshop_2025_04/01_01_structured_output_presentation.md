# Structured Output mit LLMs

## Wie strukturierte Ausgaben von LLMs funktionieren und genutzt werden können

---

## Was sind strukturierte Ausgaben?

* Formatierte Daten in einem vorhersehbaren Format
* Im Gegensatz zu Freitext leicht programmatisch verarbeitbar
* Typische Formate: JSON, XML, YAML, CSV, etc.
* Ermöglichen Validierung, Extraktion und Weiterverarbeitung

---

## Warum strukturierte Ausgaben wichtig sind

* **Interoperabilität:** Nahtlose Integration in bestehende Systeme
* **Konsistenz:** Vorhersehbare Formate für Downstream-Prozesse
* **Validierung:** Überprüfung auf Vollständigkeit und Korrektheit
* **Datenextraktion:** Gezielter Zugriff auf spezifische Informationen

---

## Grundlegende Ansätze für strukturierte Ausgaben

### 1. Anleitung im Prompt

```
Bitte gib deine Antwort im folgenden JSON-Format zurück:
{
  "name": "Produktname",
  "rating": Bewertung (1-5),
  "pros": ["Vorteil 1", "Vorteil 2"],
  "cons": ["Nachteil 1", "Nachteil 2"]
}
```

### 2. Strukturierte Parser in LangChain

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
```

---

## StrOutputParser: Der einfachste Weg

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from helpers import llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Analyst für Produktdaten."),
    ("human", "Gib Daten zu {produkt} im JSON-Format zurück.")
])

chain = prompt | llm() | StrOutputParser()
result = chain.invoke({"produkt": "iPhone 15"})
```

---

## Mit Pydantic-Modellen arbeiten

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class ProductAnalysis(BaseModel):
    name: str = Field(description="Der Name des Produkts")
    rating: float = Field(description="Bewertung von 1-5")
    pros: List[str] = Field(description="Liste von Vorteilen")
    cons: List[str] = Field(description="Liste von Nachteilen")


parser = PydanticOutputParser(pydantic_object=ProductAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Produktanalyst."),
    ("human", f"Analysiere {'{produkt}'}. {parser.get_format_instructions()}")
])

chain = prompt | llm() | parser
result = chain.invoke({"produkt": "iPhone 15"})
print(f"Name: {result.name}, Rating: {result.rating}")
```

---

## Vorteile von Pydantic

* **Typsicherheit**: Validierung der Ausgabetypen
* **Automatische Dokumentation**: Format-Anweisungen werden generiert
* **Fehlerbehandlung**: Robuste Fehlerbehandlung bei falschem Format
* **IDE-Support**: Autocompletion und Typprüfung
* **Verschachtelte Strukturen**: Unterstützung für komplexe Datenmodelle

---

## Beispiel: Sentiment-Analyse mit dem LangChain Hub

```python
from langchain import hub
from helpers import llm
from langchain.schema import StrOutputParser

sentiment_prompt = hub.pull("borislove/customer-sentiment-analysis")
client_letter = """Ich bin von dem Produkt zutiefst enttäuscht..."""

format_instructions = """Klassifiziere in JSON-Format mit:
{
  "bewertung": Zahl zwischen 1-5,
  "stimmung": "positiv" | "neutral" | "negativ",
  "kritikpunkte": ["Punkt 1", "Punkt 2"]
}"""

chain = sentiment_prompt | llm() | StrOutputParser()
result = chain.invoke({
    "client_letter": client_letter,
    "format_instructions": format_instructions
})
```

---

## Best Practices für strukturierte Ausgaben

1. **Klare Anweisungen**: Präzise Format-Spezifikation
2. **Beispiele geben**: Musterbeispiele im Prompt
3. **Temperatur senken**: Niedrige Temperatur (0-0.2) für konsistentere
   Ergebnisse
4. **Fehlerbehandlung**: Try-Except für die Verarbeitung
5. **Fallback-Mechanismen**: Alternative Strategien bei Parsing-Fehlern
6. **Format-Komplexität begrenzen**: Einfachere Strukturen sind zuverlässiger

---

## Häufige Herausforderungen

* LLM ignoriert das gewünschte Format → klarere Anweisungen
* Inkonsistente Ausgaben → niedrigere Temperatur
* Falsch formatierte JSON-Strings → Parser mit Fehlerbehebung
* Komplexe verschachtelte Strukturen → vereinfachen oder aufteilen
* Zu strenge Validierungsregeln → Flexibilität erhöhen

---

## Praktisches Beispiel: Produktbewertungsanalyse

```python
product_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """Du bist ein Experte für Produktbewertungen. 
    Formatiere deine Antwort als JSON mit:
    {
        "produktname": "Name",
        "gesamtbewertung": Zahl zwischen 1-5,
        "positive_punkte": ["Punkt 1", "Punkt 2"],
        "negative_punkte": ["Punkt 1", "Punkt 2"]
    }
    """),
    ("human", "Analysiere diese Bewertung: {review}"),
])

review_chain = product_review_prompt | llm(temperature=0.1) | StrOutputParser()
```

---

## Demo: Strukturierte Ausgabe in Aktion

* Freitext-Prompt vs. strukturierter Prompt
* Verarbeitung der strukturierten Daten
* Fehlerfälle und Problemlösungen

---

## Zusammenfassung

* Strukturierte Ausgaben sind essentiell für robuste KI-Anwendungen
* LangChain bietet verschiedene Werkzeuge (Parser)
* Pydantic-Modelle bieten Typsicherheit und Validierung
* Klare Anweisungen und Beispiele verbessern die Zuverlässigkeit
* Best Practices reduzieren Fehler und verbessern die Konsistenz

---

## Fragen?

* [LangChain Dokumentation zu Output Parsern](https://python.langchain.com/docs/modules/model_io/output_parsers/)
* [Pydantic Dokumentation](https://docs.pydantic.dev/)
