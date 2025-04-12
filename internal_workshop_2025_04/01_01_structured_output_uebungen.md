# Übungen: Structured Output mit LLMs

Diese Übungen helfen Ihnen, die Konzepte der strukturierten Ausgabe mit Large Language Models praktisch anzuwenden. Alle Übungen können im Jupyter Notebook `01_basics.ipynb` durchgeführt werden.

## Übung 1: Strukturierte Ausgabe im JSON-Format

### Aufgabe:

Erstellen Sie einen Prompt, der das LLM anweist, eine Bewertung eines Restaurants in einem strukturierten JSON-Format zurückzugeben.

### Schritte:

1. Verwenden Sie ein `ChatPromptTemplate` mit System- und Human-Message
2. Geben Sie genaue Anweisungen zum gewünschten JSON-Format
3. Definieren Sie mindestens folgende Felder:
   - `restaurant_name`: Name des Restaurants
   - `rating`: Bewertung auf einer Skala von 1-5
   - `pros`: Liste der positiven Aspekte
   - `cons`: Liste der negativen Aspekte
   - `summary`: Kurze Zusammenfassung in einem Satz
4. Testen Sie den Prompt mit mindestens zwei verschiedenen Restaurants

### Beispiel-Starter-Code:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LLM initialisieren
llm = ChatOpenAI(temperature=0.7)

# Ihr Prompt-Template hier erstellen
restaurant_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Du bist ein erfahrener Restaurant-Kritiker. Erstelle eine strukturierte Bewertung im JSON-Format für das vom Benutzer genannte Restaurant.
    
    Das JSON sollte folgendes Format haben:
    {
        "restaurant_name": "Name des Restaurants",
        "rating": Bewertung zwischen 1 und 5,
        "pros": ["positiver Aspekt 1", "positiver Aspekt 2", ...],
        "cons": ["negativer Aspekt 1", "negativer Aspekt 2", ...],
        "summary": "Zusammenfassender Satz zur Bewertung"
    }
    
    Achte auf eine korrekte JSON-Struktur ohne zusätzlichen Text vor oder nach dem JSON.
    """),
    ("human", "Bitte bewerte das Restaurant: {restaurant_name}, bekannt für {bekannt_fuer}")
])

# Chain erstellen
review_chain = restaurant_review_prompt | llm | StrOutputParser()

# Testen Sie Ihren Prompt
test_result = review_chain.invoke({
    "restaurant_name": "Zum goldenen Löwen", 
    "bekannt_fuer": "traditionelle fränkische Küche und hausgebrautes Bier"
})

print(test_result)

# Führen Sie einen weiteren Test mit einem anderen Restaurant durch
# ...
```

### Erweiterung:

Nachdem Sie die Grundfunktionalität implementiert haben, ergänzen Sie:
- Ein zusätzliches Feld `price_category` mit den möglichen Werten "günstig", "mittel" oder "teuer"
- Eine Liste mit empfohlenen Gerichten als Feld `recommended_dishes`
- Verwenden Sie `langchain.output_parsers.JsonOutputParser` anstelle von `StrOutputParser`, um das JSON direkt zu parsen

---

## Übung 2: Formatierte Textausgabe mit XML-Tags

### Aufgabe:

Erstellen Sie einen Prompt, der eine Liste von Nachrichtenartikeln in einem strukturierten XML-Format generiert.

### Schritte:

1. Erstellen Sie ein `ChatPromptTemplate`, das ein Nachrichtenthema als Input nimmt
2. Bitten Sie das LLM, drei kurze Nachrichtenartikel zu diesem Thema zu erstellen
3. Die Ausgabe soll im XML-Format mit folgenden Tags sein:
   - `<articles>` als Wurzelelement
   - `<article>` für jeden Artikel mit dem Attribut `id`
   - `<headline>` für die Überschrift
   - `<date>` für das Datum (aktuell oder fiktiv)
   - `<content>` für den Inhalt (kurzer Absatz)
   - `<source>` für die Quelle (fiktive Nachrichtenquelle)
4. Testen Sie den Prompt mit verschiedenen Nachrichtenthemen

### Beispiel-Starter-Code:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LLM initialisieren
llm = ChatOpenAI(temperature=0.7)

# Ihr Prompt-Template hier erstellen
news_articles_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Du bist ein erfahrener Journalist. Erstelle drei kurze Nachrichtenartikel zum angegebenen Thema.
    
    Die Artikel sollen im folgenden XML-Format zurückgegeben werden:
    
    <articles>
      <article id="1">
        <headline>Titel des ersten Artikels</headline>
        <date>YYYY-MM-DD</date>
        <content>Kurzer Absatz mit dem Inhalt des Artikels.</content>
        <source>Name einer fiktiven Nachrichtenquelle</source>
      </article>
      <!-- Weitere Artikel mit id="2" und id="3" -->
    </articles>
    
    Achte auf eine korrekte XML-Struktur ohne zusätzlichen Text vor oder nach dem XML.
    """),
    ("human", "Bitte erstelle Nachrichtenartikel zum Thema: {thema}")
])

# Chain erstellen
news_chain = news_articles_prompt | llm | StrOutputParser()

# Testen Sie Ihren Prompt
test_result = news_chain.invoke({"thema": "Neue Entwicklungen in der künstlichen Intelligenz"})

print(test_result)

# Führen Sie einen weiteren Test mit einem anderen Thema durch
# ...
```

### Erweiterung:

- Fügen Sie ein `<category>` Tag hinzu, das die Artikelkategorie angibt (z.B. Politik, Wirtschaft, Technologie)
- Ergänzen Sie ein `<keywords>` Tag mit einer Liste von Schlüsselwörtern
- Implementieren Sie einen einfachen XML-Parser in Python, der die Artikel als Dictionary extrahiert

---

## Übung 3: Sentiment-Analyse mit strukturierter Ausgabe

### Aufgabe:

Implementieren Sie eine Sentiment-Analyse-Funktion, die Kundenfeedback analysiert und strukturierte Ergebnisse zurückgibt.

### Schritte:

1. Verwenden Sie den im Kurs erwähnten Sentiment-Analyse-Prompt aus dem LangChain Hub oder erstellen Sie einen eigenen
2. Die Ausgabe soll folgende Informationen enthalten:
   - Stimmung (positiv, neutral, negativ) mit einem numerischen Wert (-1.0 bis 1.0)
   - Wichtigste Schlüsselwörter, die die Stimmung beeinflussen
   - Eine kurze Begründung für die Stimmungsbewertung
   - Handlungsempfehlungen basierend auf dem Feedback
3. Testen Sie die Funktion mit verschiedenen Kundenfeedbacks (positiv, neutral und negativ)

### Beispiel-Starter-Code:

```python
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LLM initialisieren
llm = ChatOpenAI(temperature=0)

# Option 1: Laden des Prompts aus dem Hub
try:
    # Sentiment-Analyse-Prompt vom Hub laden
    sentiment_prompt = hub.pull("borislove/customer-sentiment-analysis")
except:
    # Falls der Hub nicht verfügbar ist, eigenen Prompt definieren
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Du bist ein Experte für Sentiment-Analyse von Kundenfeedback.
        Analysiere das gegebene Kundenfeedback und erstelle eine strukturierte Ausgabe mit folgendem Format:
        
        STIMMUNGSWERT: [Numerischer Wert zwischen -1.0 (sehr negativ) und 1.0 (sehr positiv)]
        STIMMUNGSKATEGORIE: [positiv/neutral/negativ]
        SCHLÜSSELWÖRTER: [Liste der wichtigsten stimmungsbezogenen Wörter]
        BEGRÜNDUNG: [Kurze Erklärung für die Stimmungsbewertung]
        EMPFEHLUNGEN: [Vorschläge für Maßnahmen basierend auf dem Feedback]
        
        {format_instructions}
        """),
        ("human", "{client_letter}")
    ])

# Chain erstellen
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# Testfälle
positive_feedback = """
Ich bin begeistert von Ihrem Produkt! Die Benutzeroberfläche ist intuitiv und die Funktionen sind genau das, was ich brauche. Der Kundendienst hat mir bei einer Frage sehr schnell geholfen. Werde Sie definitiv weiterempfehlen!
"""

negative_feedback = """
Leider bin ich sehr enttäuscht von Ihrem Service. Seit drei Tagen versuche ich, jemanden zu erreichen, aber niemand antwortet. Das Produkt funktioniert nicht wie beschrieben und die Installation war viel komplizierter als angegeben. Ich überlege, das Produkt zurückzugeben.
"""

neutral_feedback = """
Das Produkt erfüllt seinen Zweck. Es gibt einige nützliche Funktionen, aber auch ein paar Dinge, die verbessert werden könnten. Die Lieferung kam pünktlich an. Insgesamt eine durchschnittliche Erfahrung.
"""

# Testen Sie die Sentiment-Analyse mit verschiedenen Feedbacks
format_instructions = """
Zusätzlich zur strukturierten Ausgabe sollst du den Text grammatikalisch korrekt und gut formatiert gestalten.
"""

print("Positive Feedback Analyse:")
print(sentiment_chain.invoke({"client_letter": positive_feedback, "format_instructions": format_instructions}))
print("\n" + "-"*50 + "\n")

# Führen Sie ähnliche Tests für die anderen Feedback-Typen durch
```

### Erweiterung:

- Erweitern Sie den Prompt um eine Kategorisierung des Feedbacks (z.B. Produkt, Service, Preisgestaltung, Lieferung)
- Implementieren Sie eine Funktion, die mehrere Feedback-Texte analysiert und eine zusammenfassende Statistik erstellt
- Experimentieren Sie mit verschiedenen Temperaturen für das LLM und beobachten Sie die Auswirkungen auf die Ergebnisse

---

## Übung 4: Strukturierte Produktbeschreibungen

### Aufgabe:

Erstellen Sie einen Prompt, der strukturierte Produktbeschreibungen in einem konsistenten Format für einen E-Commerce-Shop generiert.

### Schritte:

1. Verwenden Sie `ChatPromptTemplate` mit System- und Human-Message
2. Das System soll strukturierte Produktbeschreibungen für verschiedene Produktkategorien generieren
3. Die strukturierte Ausgabe soll folgendes Format haben:
   ```
   ### [Produktname]
   
   **Kategorie:** [Produktkategorie]
   
   **Preis:** [Preisbereich]
   
   **Hauptmerkmale:**
   - [Merkmal 1]
   - [Merkmal 2]
   - [Merkmal 3]
   
   **Beschreibung:**
   [Absatz mit detaillierter Produktbeschreibung]
   
   **Ideal für:**
   - [Zielgruppe 1]
   - [Zielgruppe 2]
   
   **Technische Details:**
   | Eigenschaft | Wert |
   |------------|------|
   | [Eigenschaft 1] | [Wert 1] |
   | [Eigenschaft 2] | [Wert 2] |
   | [Eigenschaft 3] | [Wert 3] |
   ```
4. Testen Sie den Prompt mit mindestens zwei verschiedenen Produkten

### Beispiel-Starter-Code:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LLM initialisieren
llm = ChatOpenAI(temperature=0.7)

# Ihr Prompt-Template hier erstellen
product_description_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Du bist ein erfahrener Produktbeschreibungs-Writer für einen E-Commerce-Shop.
    Erstelle eine strukturierte Produktbeschreibung für das vom Benutzer genannte Produkt.
    
    Verwende das folgende Markdown-Format:
    
    ### [Produktname]
    
    **Kategorie:** [Produktkategorie]
    
    **Preis:** [Preisbereich]
    
    **Hauptmerkmale:**
    - [Merkmal 1]
    - [Merkmal 2]
    - [Merkmal 3]
    
    **Beschreibung:**
    [Absatz mit detaillierter Produktbeschreibung]
    
    **Ideal für:**
    - [Zielgruppe 1]
    - [Zielgruppe 2]
    
    **Technische Details:**
    | Eigenschaft | Wert |
    |------------|------|
    | [Eigenschaft 1] | [Wert 1] |
    | [Eigenschaft 2] | [Wert 2] |
    | [Eigenschaft 3] | [Wert 3] |
    
    Erstelle eine überzeugende, aber ehrliche Beschreibung mit relevanten technischen Details für die jeweilige Produktkategorie.
    """),
    ("human", "Bitte erstelle eine Produktbeschreibung für: {produktname} in der Kategorie {kategorie}")
])

# Chain erstellen
description_chain = product_description_prompt | llm | StrOutputParser()

# Testen Sie Ihren Prompt
test_result = description_chain.invoke({
    "produktname": "UltraSound X500 Bluetooth-Kopfhörer", 
    "kategorie": "Audio & HiFi"
})

print(test_result)

# Führen Sie einen weiteren Test mit einem anderen Produkt durch
test_result2 = description_chain.invoke({
    "produktname": "FreshMaster 3000 Standmixer", 
    "kategorie": "Küchengeräte"
})

print(test_result2)
```

### Erweiterung:

- Fügen Sie ein Bewertungssystem mit Sternen (1-5) für verschiedene Aspekte des Produkts hinzu
- Ergänzen Sie einen Abschnitt mit Vergleichen zu ähnlichen Produkten
- Implementieren Sie eine Version, die direkt HTML-Code für eine Webseite generiert

---

## Bonusübung: Komplexe strukturierte Ausgabe mit mehreren Formaten

### Aufgabe:

Erstellen Sie einen Prompt, der Daten zu einer fiktiven Person generiert und in mehreren Formaten zurückgibt: JSON für technische Daten, Markdown für eine menschenlesbare Biographie und einem CSV-Schnipsel für Datenbankimport.

### Schritte:

1. Erstellen Sie einen Prompt, der demografische Informationen zu einer fiktiven Person generiert
2. Die Ausgabe soll folgende Teile enthalten:
   - Ein JSON-Objekt mit Grunddaten (Name, Alter, Beruf, Wohnort, etc.)
   - Eine Markdown-formatierte Biographie mit Überschriften und Absätzen
   - Eine CSV-Zeile mit den wichtigsten Daten im Format für Datenbankimport
3. Verwenden Sie `langchain_core.output_parsers.structured.ResponseSchema` und `StructuredOutputParser` für die strukturierte Ausgabe

### Beispiel-Starter-Code:

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.structured import ResponseSchema, StructuredOutputParser

# LLM initialisieren
llm = ChatOpenAI(temperature=0.8)

# Definieren der Ausgabestruktur mit ResponseSchema
response_schemas = [
    ResponseSchema(name="json_data", description="JSON-Objekt mit den Grunddaten der Person"),
    ResponseSchema(name="markdown_bio", description="Markdown-formatierte Biographie"),
    ResponseSchema(name="csv_data", description="CSV-Zeile für Datenbankimport")
]

# Parser erstellen
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Format-Anweisungen bekommen
format_instructions = parser.get_format_instructions()

# Prompt erstellen
person_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Du bist ein Experte für die Generierung fiktiver Personendaten. 
    Erstelle basierend auf den angegebenen Kriterien eine detaillierte fiktive Person.
    
    {format_instructions}
    """),
    ("human", """
    Erstelle eine fiktive Person mit folgenden Attributen:
    - Geschlecht: {geschlecht}
    - Altersbereich: {alter}
    - Berufskategorie: {beruf}
    - Wohnort: {wohnort}
    
    Füge weitere passende Details hinzu, um die Person realistisch wirken zu lassen.
    """)
])

# Chain erstellen
person_chain = person_generator_prompt | llm | parser

# Testen Sie Ihren Prompt
test_result = person_chain.invoke({
    "format_instructions": format_instructions,
    "geschlecht": "weiblich",
    "alter": "30-40",
    "beruf": "Technologie",
    "wohnort": "München"
})

print("JSON-Daten:")
print(test_result["json_data"])
print("\nMarkdown-Biographie:")
print(test_result["markdown_bio"])
print("\nCSV-Daten:")
print(test_result["csv_data"])

# Führen Sie einen weiteren Test mit anderen Parametern durch
```

Diese Bonusübung kombiniert verschiedene Ausgabeformate und ist anspruchsvoller als die Grundübungen.

---

Die oben genannten Übungen decken das Thema "Structured Output" ab und bauen auf die im Workshop behandelten Konzepte auf. Die Übungen sind so gestaltet, dass sie in Jupyter Notebooks ausgeführt werden können und bieten verschiedene Schwierigkeitsstufen mit Erweiterungsmöglichkeiten.
