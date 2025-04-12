# Structured Output mit LLMs in LangChain

LLMs generieren standardmäßig Freitext, aber für viele Anwendungen benötigen wir
strukturierte Daten, die programmatisch verarbeitet werden können. In diesem
Teil des Workshops lernen Sie verschiedene Techniken kennen, um LLMs dazu zu
bringen, strukturierte Ausgaben wie JSON, XML oder andere spezifische Formate zu
erzeugen.

## Grundlagen der strukturierten Ausgabe

Strukturierte Ausgaben von LLMs haben folgende Vorteile:

1. **Konsistenz:** Vorhersehbare Formate für nachgelagerte Verarbeitung
2. **Programmatische Verarbeitung:** Einfache Weiterverarbeitung in Anwendungen
3. **Validierung:** Möglichkeit zur Überprüfung der Ausgabestruktur
4. **Integration:** Nahtlose Einbindung in bestehende Systeme

## Methoden für strukturierte Ausgaben

### 1. Einfache Textformatierung mit StrOutputParser

Der einfachste Ansatz ist, das LLM durch Anweisungen im Prompt zur Formatierung
zu bringen:

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from helpers import llm

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein hilfsbereicher Assistent, der Daten im JSON-Format zurückgibt."),
    ("human",
     "Gib mir Informationen über die Stadt Berlin im JSON-Format mit den Feldern 'name', 'land', 'einwohner' und einem Array 'sehenswuerdigkeiten'.")
])

chain = prompt | llm() | StrOutputParser()
result = chain.invoke({})
print(result)
```

### 2. Strukturierte Ausgabe mit Pydantic-Modellen

Fortgeschrittener und zuverlässiger ist die Verwendung von Pydantic-Modellen zur
Definition der erwarteten Struktur und eines speziellen OutputParsers:

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Definition des Ausgabeformats als Pydantic-Modell
class Stadt(BaseModel):
    name: str = Field(description="Der Name der Stadt")
    land: str = Field(description="Das Land, in dem die Stadt liegt")
    einwohner: int = Field(description="Die Anzahl der Einwohner")
    sehenswuerdigkeiten: List[str] = Field(
        description="Liste bekannter Sehenswürdigkeiten")


# Parser erstellen
parser = PydanticOutputParser(pydantic_object=Stadt)

# Prompt mit Parser-Anweisungen
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein hilfsbereicher Assistent, der Informationen in strukturierter Form zurückgibt."),
    ("human",
     f"Gib mir Informationen über die Stadt Berlin. {parser.get_format_instructions()}"),
])

# Chain erstellen
chain = prompt | llm() | parser
result = chain.invoke({})
print(result)
print(f"Name: {result.name}")
print(f"Land: {result.land}")
print(f"Einwohner: {result.einwohner}")
print(f"Sehenswürdigkeiten: {', '.join(result.sehenswuerdigkeiten)}")
```

## Praktische Übung

### Aufgabe 1: Anpassung eines Sentiment-Analyse-Prompts

Modifizieren Sie den Sentiment-Analyse-Prompt, um neben der Stimmungsbewertung
auch konkrete Verbesserungsvorschläge zu generieren:

```python
from langchain import hub
from helpers import llm
from langchain.schema import StrOutputParser

sentiment_prompt = hub.pull("borislove/customer-sentiment-analysis")
client_letter = """Ich bin von dem Volleyballschläger zutiefst enttäuscht. Zuerst ist der Griff abgefallen, danach auch noch der Dynamo. Außerdem riecht er noch schlechter als er schmeckt. Wieso ist das immer so ein Ärger mit euch?"""

# Anpassen der Format-Anweisungen für erweiterte Antworten
format_instructions = """Zusätzlich zur numerischen Klassifizierung sollst du:
1. Die konkreten Kritikpunkte in Stichpunkten zusammenfassen
2. Einen Vorschlag machen, was dem Kunden geantwortet werden sollte
3. Drei konkrete Maßnahmen zur Produktverbesserung empfehlen

Formatiere die Ausgabe in übersichtlicher Form mit Überschriften und Aufzählungen.
"""

sentiment_chain = sentiment_prompt | llm() | StrOutputParser()
result = sentiment_chain.invoke({"client_letter": client_letter,
                                 "format_instructions": format_instructions})
print(result)
```

### Aufgabe 2: Erstellen eines strukturierten Outputs für Produktbewertungen

Entwickeln Sie ein Prompt-Template, das Kundenbewertungen in eine strukturierte
Form bringt:

```python
from langchain.prompts import ChatPromptTemplate
from helpers import llm
from langchain.schema import StrOutputParser

product_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """Du bist ein Experte für die Analyse von Produktbewertungen. 
    Extrahiere Informationen aus Kundenbewertungen und gib sie im folgenden JSON-Format zurück:
    {
        "produktname": "Name des Produkts",
        "gesamtbewertung": Zahl zwischen 1-5,
        "positive_punkte": ["Liste positiver Aspekte"],
        "negative_punkte": ["Liste negativer Aspekte"],
        "verbesserungsvorschlaege": ["Liste von Verbesserungsvorschlägen"]
    }
    """
     ),
    ("human", "Analysiere folgende Produktbewertung: {review}"),
])

# Beispielbewertung
review = """Der Kaffeevollautomat XYZ-5000 hat mich größtenteils überzeugt. 
Die Bedienung ist super einfach und intuitiv, das Design passt gut in meine Küche. 
Der Kaffee schmeckt ausgezeichnet und hat eine gute Crema. 
Allerdings ist die Maschine sehr laut beim Mahlen und die Reinigung des Milchsystems ist umständlich. 
Die App-Steuerung stürzt manchmal ab. Ich würde mir ein leiseres Mahlwerk und ein einfacheres Reinigungssystem wünschen."""

review_chain = product_review_prompt | llm() | StrOutputParser()
result = review_chain.invoke({"review": review})
print(result)
```

## Tipps für effektive strukturierte Ausgaben

1. **Klare Anweisungen**: Je genauer die Format-Anweisungen, desto konsistenter
   die Ausgabe
2. **Beispiel mitliefern**: Ein Beispiel für das gewünschte Format erhöht die
   Wahrscheinlichkeit korrekter Ausgaben
3. **Fehlerbehandlung**: Implementieren Sie Validierung für die erhaltenen
   strukturierten Daten
4. **Iteratives Testen**: Optimieren Sie Prompts durch wiederholtes Testen mit
   verschiedenen Eingaben
5. **Niedrige Temperatur**: Für strukturierte Ausgaben empfiehlt sich eine
   niedrige Temperatur (z.B. 0-0.2)

## Weiterführende Ressourcen

- [LangChain Dokumentation zu Output Parsern](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [LangChain Hub](https://smith.langchain.com/hub) für vorgefertigte Prompts
- [Pydantic Dokumentation](https://docs.pydantic.dev/) für strukturierte
  Datenmodelle
