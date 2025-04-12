# Chaining: Übungen und Lösungen

## Übung 1: Einfache sequenzielle Chain

**Aufgabe**: Erstellen Sie eine einfache Chain, die einen Text zusammenfasst und
anschließend in eine andere Sprache übersetzt.

**Lösungsansatz**:

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from helpers import llm

# Prompts definieren
zusammenfassung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein Experte für Textzusammenfassungen. Fasse den folgenden Text in maximal 3 Sätzen zusammen."),
    ("human", "{text}")
])

uebersetzung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein professioneller Übersetzer. Übersetze den folgenden Text ins {zielsprache}."),
    ("human", "{text}")
])

# Chains erstellen
zusammenfassen_chain = zusammenfassung_prompt | llm() | StrOutputParser()
uebersetzen_chain = uebersetzung_prompt | llm() | StrOutputParser()


# Chains verketten
def chain_zusammenfassung_uebersetzung(text, zielsprache):
    # Erst zusammenfassen
    zusammenfassung = zusammenfassen_chain.invoke({"text": text})

    # Dann übersetzen
    uebersetzung = uebersetzen_chain.invoke({
        "text": zusammenfassung,
        "zielsprache": zielsprache
    })

    return {
        "original": text,
        "zusammenfassung": zusammenfassung,
        "uebersetzung": uebersetzung
    }


# Beispiel-Verwendung
beispieltext = """
Die künstliche Intelligenz hat in den letzten Jahren eine rasante Entwicklung erlebt. 
Large Language Models wie GPT-4 können heute komplexe Texte verfassen, Programmcode schreiben und 
sogar Bilder generieren. Diese Fortschritte eröffnen neue Möglichkeiten in vielen Bereichen, 
von der Automatisierung alltäglicher Aufgaben bis hin zur Unterstützung kreativer Prozesse. 
Gleichzeitig werfen sie wichtige Fragen zu Ethik, Datenschutz und dem zukünftigen Zusammenspiel 
von Mensch und Maschine auf.
"""

ergebnis = chain_zusammenfassung_uebersetzung(beispieltext, "Englisch")
print(f"Zusammenfassung: {ergebnis['zusammenfassung']}")
print(f"\nÜbersetzung: {ergebnis['uebersetzung']}")
```

### Fortgeschrittene Lösung mit LCEL

```python
# Alternative Implementierung mit LCEL
komplexe_chain = (
    # Input-Transformation
        {
            "text": lambda x: x["original_text"],
            "zielsprache": lambda x: x["zielsprache"]
        }
        # Zusammenfassung
        | {
            "zusammenfassung": zusammenfassen_chain,
            "zielsprache": lambda x: x["zielsprache"]
        }
        # Übersetzung
        | {
            "zusammenfassung": lambda x: x["zusammenfassung"],
            "uebersetzung": lambda x: uebersetzen_chain.invoke({
                "text": x["zusammenfassung"],
                "zielsprache": x["zielsprache"]
            })
        }
)

ergebnis_lcel = komplexe_chain.invoke({
    "original_text": beispieltext,
    "zielsprache": "Französisch"
})
```

## Übung 2: Parallele Verarbeitung mit RunnableMap

**Aufgabe**: Erstellen Sie eine Chain, die einen Text parallel in drei
verschiedene Richtungen analysiert: Stimmung, Hauptthemen und Lesbarkeitsniveau.

**Lösungsansatz**:

```python
from langchain.schema.runnable import RunnableMap
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Ausgabemodelle definieren
class Stimmungsanalyse(BaseModel):
    stimmung: str = Field(
        description="Die allgemeine Stimmung des Textes (positiv, neutral, negativ)")
    intensität: int = Field(
        description="Stimmungsintensität von 1 (schwach) bis 10 (stark)")


class Themenanalyse(BaseModel):
    hauptthemen: List[str] = Field(
        description="Die drei wichtigsten Themen im Text")
    schlüsselwörter: List[str] = Field(
        description="Die fünf wichtigsten Schlüsselwörter")


class Lesbarkeitsanalyse(BaseModel):
    niveau: str = Field(
        description="Lesbarkeitsniveau (einfach, mittel, komplex)")
    zielgruppe: str = Field(description="Geeignete Zielgruppe für diesen Text")
    verbesserungsvorschläge: List[str] = Field(
        description="Zwei Vorschläge zur Verbesserung der Lesbarkeit")


# Parser erstellen
stimmung_parser = PydanticOutputParser(pydantic_object=Stimmungsanalyse)
themen_parser = PydanticOutputParser(pydantic_object=Themenanalyse)
lesbarkeit_parser = PydanticOutputParser(pydantic_object=Lesbarkeitsanalyse)

# Prompts erstellen
stimmung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Führe eine Stimmungsanalyse des folgenden Textes durch.\n\n{format_instructions}"),
    ("human", "{text}")
])

themen_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Analysiere die Hauptthemen des folgenden Textes.\n\n{format_instructions}"),
    ("human", "{text}")
])

lesbarkeit_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Bewerte die Lesbarkeit des folgenden Textes.\n\n{format_instructions}"),
    ("human", "{text}")
])

# Einzelne Chains
stimmung_chain = stimmung_prompt.partial(
    format_instructions=stimmung_parser.get_format_instructions()) | llm() | stimmung_parser
themen_chain = themen_prompt.partial(
    format_instructions=themen_parser.get_format_instructions()) | llm() | themen_parser
lesbarkeit_chain = lesbarkeit_prompt.partial(
    format_instructions=lesbarkeit_parser.get_format_instructions()) | llm() | lesbarkeit_parser

# Parallele Chain mit RunnableMap
parallel_analyse = RunnableMap({
    "stimmung": lambda x: stimmung_chain.invoke({"text": x["text"]}),
    "themen": lambda x: themen_chain.invoke({"text": x["text"]}),
    "lesbarkeit": lambda x: lesbarkeit_chain.invoke({"text": x["text"]})
})

# Test mit einem Beispieltext
beispieltext = """
Die neuen Richtlinien zur Reduzierung des CO2-Ausstoßes wurden heute vom Umweltministerium veröffentlicht. 
Sie sehen vor, dass Unternehmen bis 2030 ihren Kohlenstoffausstoß um 55% im Vergleich zu 1990 senken müssen. 
Kritiker bemängeln die hohen Kosten und den engen Zeitrahmen, während Befürworter die Maßnahmen als längst 
überfällig betrachten. Experten gehen davon aus, dass besonders die Energiebranche und die Automobilindustrie 
vor großen Herausforderungen stehen werden.
"""

ergebnisse = parallel_analyse.invoke({"text": beispieltext})

print("Stimmungsanalyse:")
print(f"Stimmung: {ergebnisse['stimmung'].stimmung}")
print(f"Intensität: {ergebnisse['stimmung'].intensität}/10")

print("\nThemenanalyse:")
print(f"Hauptthemen: {ergebnisse['themen'].hauptthemen}")
print(f"Schlüsselwörter: {ergebnisse['themen'].schlüsselwörter}")

print("\nLesbarkeitsanalyse:")
print(f"Niveau: {ergebnisse['lesbarkeit'].niveau}")
print(f"Zielgruppe: {ergebnisse['lesbarkeit'].zielgruppe}")
print("Verbesserungsvorschläge:")
for vorschlag in ergebnisse['lesbarkeit'].verbesserungsvorschläge:
    print(f"- {vorschlag}")
```

## Übung 3: Streaming und Fortschrittsanzeige

**Aufgabe**: Implementieren Sie eine Chain, die beim Generieren eines längeren
Textes den Fortschritt streamt und visualisiert.

**Lösungsansatz**:

```python
import asyncio
import time
from IPython.display import clear_output


async def visualisiere_textgenerierung():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Du bist ein brillanter Geschichtenerzähler. Schreibe eine fesselnde Kurzgeschichte zum Thema {thema} mit etwa 300 Wörtern."),
        ("human", "Mein Thema ist: {thema}")
    ])

    chain = prompt | llm() | StrOutputParser()

    # Fortschrittsanzeige vorbereiten
    gesamter_text = ""
    start_zeit = time.time()
    wortanzahl = 0

    print("Generiere Geschichte...")
    print("=" * 50)

    async for chunk in chain.astream(
            {"thema": "Eine Begegnung mit künstlicher Intelligenz"}):
        gesamter_text += chunk
        neue_wortanzahl = len(gesamter_text.split())

        if neue_wortanzahl > wortanzahl:
            wortanzahl = neue_wortanzahl
            verstrichene_zeit = time.time() - start_zeit

            # Fortschrittsanzeige aktualisieren
            clear_output(wait=True)
            print(
                f"Generiere Geschichte... ({wortanzahl} Wörter, {verstrichene_zeit:.1f} Sekunden)")
            print("=" * 50)
            print(gesamter_text)

        # Kleine Pause für die Visualisierung
        await asyncio.sleep(0.01)

    gesamtzeit = time.time() - start_zeit
    print("\n" + "=" * 50)
    print(
        f"Fertig! Generiert in {gesamtzeit:.2f} Sekunden ({wortanzahl} Wörter, {wortanzahl / gesamtzeit:.1f} Wörter/Sekunde)")

# In einem Jupyter Notebook ausführen mit:
# await visualisiere_textgenerierung()
```

## Übung 4: Dynamische Chain mit Entscheidungslogik

**Aufgabe**: Erstellen Sie eine Chain, die basierend auf einer Klassifikation
des Eingabetextes unterschiedliche Verarbeitungswege wählt.

**Lösungsansatz**:

```python
from langchain.schema.runnable import RunnableBranch

# Klassifikator für den Texttyp
klassifikator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Klassifiziere den folgenden Text als eine der Kategorien: 'FRAGE', 'BERICHT', 'MEINUNG' oder 'ANWEISUNG'."),
    ("human", "{text}")
])

klassifikator_chain = klassifikator_prompt | llm() | StrOutputParser()

# Spezifische Chains für verschiedene Texttypen
frage_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein hilfreicher Assistent, der sachlich und präzise auf Fragen antwortet."),
    ("human", "{text}")
])

bericht_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein Analyst, der Berichte prägnant zusammenfasst und wichtige Erkenntnisse hervorhebt."),
    ("human",
     "Fasse den folgenden Bericht zusammen und hebe wichtige Punkte hervor:\n\n{text}")
])

meinung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein ausgewogener Kommentator, der verschiedene Perspektiven zu einem Meinungstext aufzeigt."),
    (
    "human", "Zeige verschiedene Perspektiven zu dieser Meinung auf:\n\n{text}")
])

anweisung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein Experte darin, Anweisungen zu optimieren und in klare Schritte zu gliedern."),
    ("human",
     "Optimiere und strukturiere diese Anweisungen in klare Schritte:\n\n{text}")
])

# Chains für die spezifischen Texttypen
frage_chain = frage_prompt | llm() | StrOutputParser()
bericht_chain = bericht_prompt | llm() | StrOutputParser()
meinung_chain = meinung_prompt | llm() | StrOutputParser()
anweisung_chain = anweisung_prompt | llm() | StrOutputParser()


# Branch für die Entscheidungslogik basierend auf Klassifikation
def get_chain_for_text_type(inputs):
    text = inputs["text"]
    text_type = klassifikator_chain.invoke({"text": text}).strip().upper()

    if "FRAGE" in text_type:
        return frage_chain
    elif "BERICHT" in text_type:
        return bericht_chain
    elif "MEINUNG" in text_type:
        return meinung_chain
    elif "ANWEISUNG" in text_type:
        return anweisung_chain
    else:
        # Fallback für unbekannte Texttypen
        return lambda
            x: f"Konnte den Texttyp nicht klar klassifizieren: {text_type}. Bitte formulieren Sie Ihren Text anders."


# Dynamische Chain mit RunnableBranch
dynamische_chain = RunnableBranch(
    (lambda x: get_chain_for_text_type(x), lambda x: x["text"])
)

# Beispiele testen
texte = [
    "Wie funktionieren Large Language Models und was sind ihre Einschränkungen?",
    "Der neue Quartalsbericht zeigt einen Anstieg des Umsatzes um 12% gegenüber dem Vorjahr. Die Kosten sind jedoch um 15% gestiegen, was zu einem leichten Rückgang des Nettogewinns führte.",
    "Ich finde, dass die derzeitige Bildungspolitik zu stark auf Standardisierung setzt und zu wenig Raum für individuelle Förderung lässt.",
    "Installieren Sie die Software, indem Sie die Datei herunterladen, den Installer ausführen und den Anweisungen folgen. Starten Sie dann das Programm und konfigurieren Sie es unter Einstellungen."
]

for i, text in enumerate(texte):
    print(f"\n=== Beispiel {i + 1} ===")
    print(f"Text: {text[:50]}...")
    ergebnis = dynamische_chain.invoke({"text": text})
    print(f"\nErgebnis: {ergebnis[:100]}...")
```

## Bonus-Übung: Mehrsprachige Content-Pipeline

**Aufgabe**: Erstellen Sie eine umfassende Chain, die Content in mehreren
Sprachen generiert, zusammenfasst und für verschiedene Plattformen optimiert.

**Lösungsansatz**:

```python
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from typing import List, Dict, Any
import json

# Inhaltsgenerierung
content_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein Content-Experte, der informative und fesselnde Texte erstellt."),
    ("human",
     "Erstelle einen informativen Artikel von etwa 300 Wörtern zum Thema {thema}.")
])

# Übersetzung
übersetzung_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein professioneller Übersetzer, der Texte natürlich und idiomatisch übersetzt."),
    ("human",
     "Übersetze den folgenden Text ins {sprache}. Behalte den Stil und Ton bei.\n\nText: {text}")
])


# Plattform-Optimierung
class PlattformOptimierung(BaseModel):
    twitter: str = Field(description="Version für Twitter (max. 240 Zeichen)")
    linkedin: str = Field(
        description="Version für LinkedIn (professioneller Ton, 3-4 Sätze)")
    facebook: str = Field(
        description="Version für Facebook (informell, unterhaltsam, 2-3 kurze Absätze)")


plattform_parser = PydanticOutputParser(pydantic_object=PlattformOptimierung)

plattform_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Erstelle optimierte Versionen des folgenden Inhalts für verschiedene Social-Media-Plattformen.\n\n{format_instructions}"),
    ("human", "Original-Inhalt:\n\n{text}")
])

# Chains definieren
content_chain = content_prompt | llm() | StrOutputParser()
übersetzung_chain = übersetzung_prompt | llm() | StrOutputParser()
plattform_chain = plattform_prompt.partial(
    format_instructions=plattform_parser.get_format_instructions()) | llm() | plattform_parser


# Komplexe Content-Pipeline
def content_pipeline(thema: str, sprachen: List[str]) -> Dict[str, Any]:
    # Hauptinhalt generieren
    original_content = content_chain.invoke({"thema": thema})

    ergebnis = {
        "original": {
            "text": original_content,
            "plattformen": plattform_chain.invoke({"text": original_content})
        }
    }

    # Übersetzungen
    for sprache in sprachen:
        übersetzung = übersetzung_chain.invoke({
            "text": original_content,
            "sprache": sprache
        })

        plattform_versionen = plattform_chain.invoke({"text": übersetzung})

        ergebnis[sprache] = {
            "text": übersetzung,
            "plattformen": plattform_versionen
        }

    return ergebnis


# Beispielausführung
thema = "Die Zukunft des Remote-Arbeitens nach der Pandemie"
sprachen = ["Englisch", "Spanisch", "Französisch"]

ergebnis = content_pipeline(thema, sprachen)

# Ausgabe eines Teils des Ergebnisses
print(f"=== Original ===\n{ergebnis['original']['text'][:200]}...\n")
print(
    f"=== Twitter-Version ===\n{ergebnis['original']['plattformen'].twitter}\n")

print(
    f"=== Englische Übersetzung ===\n{ergebnis['Englisch']['text'][:200]}...\n")
print(
    f"=== LinkedIn-Version (Englisch) ===\n{ergebnis['Englisch']['plattformen'].linkedin}\n")
```

Diese Übungen decken die wichtigsten Aspekte des Chainings in LangChain ab und
zeigen, wie man von einfachen sequenziellen Chains zu komplexen, dynamischen
Workflows fortschreiten kann.
