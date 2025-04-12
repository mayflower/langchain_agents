# Übungen: Prompt-Management

## Übung 1: Strukturierte Prompts erstellen und optimieren

**Aufgabe:** Erstellen Sie verschiedene Versionen eines Prompts und vergleichen Sie deren Effizienz.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

# Initialisierung
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# 1. Erstellen Sie drei verschiedene Prompts für die gleiche Aufgabe
# Aufgabe: Zusammenfassung eines gegebenen Textes in 3 Sätzen

# Variante 1: Einfacher Prompt ohne Struktur
basic_prompt = ChatPromptTemplate.from_template(
    "Fasse den folgenden Text in 3 Sätzen zusammen: {text}"
)

# Variante 2: Prompt mit detaillierteren Anweisungen
detailed_prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein professioneller Zusammenfasser, der komplexe Texte präzise auf den Punkt bringt."),
    ("human", "Fasse den folgenden Text in genau 3 Sätzen zusammen. Achte darauf, die wichtigsten Informationen zu erfassen: {text}")
])

# Variante 3: Strukturierter Prompt mit Formatierungs-Anweisungen
structured_prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Experte für Textzusammenfassungen."),
    ("human", """
    Fasse den folgenden Text in genau 3 Sätzen zusammen:
    
    Text: {text}
    
    Gehe dabei wie folgt vor:
    1. Identifiziere die Hauptaussage
    2. Bestimme die wichtigsten unterstützenden Details
    3. Fasse diese in 3 prägnanten, informativen Sätzen zusammen
    
    Format deiner Antwort:
    Zusammenfassung: [Deine 3 Sätze hier]
    """)
])

# 2. Definieren Sie einen Beispieltext zum Testen der Prompts
beispieltext = """
Künstliche Intelligenz (KI) revolutioniert zahlreiche Branchen durch ihre Fähigkeit, menschenähnliche kognitive Funktionen zu simulieren. 
Besonders Large Language Models (LLMs) wie GPT-4 haben in den letzten Jahren erhebliche Fortschritte gemacht und können nun komplexe Texte generieren, 
Programmcode schreiben und sogar kreative Aufgaben bewältigen. Trotz dieser beeindruckenden Fähigkeiten stehen KI-Systeme vor bedeutenden Herausforderungen 
wie Halluzinationen (das Generieren falscher Informationen), ethischen Bedenken bezüglich Urheberrecht und Verzerrungen in den Trainingsdaten, 
sowie der Notwendigkeit enormer Rechenleistung für Training und Betrieb. Forscher arbeiten intensiv daran, diese Probleme zu lösen und gleichzeitig 
die Grundlagenforschung voranzutreiben, um KI-Systeme zu entwickeln, die tatsächlich verstehen, was sie verarbeiten, anstatt nur statistische Muster zu erkennen.
"""

# 3. Chains erstellen und Ergebnisse vergleichen
basic_chain = basic_prompt | llm | output_parser
detailed_chain = detailed_prompt | llm | output_parser
structured_chain = structured_prompt | llm | output_parser

# Ausführen und Ergebnisse vergleichen
basic_result = basic_chain.invoke({"text": beispieltext})
detailed_result = detailed_chain.invoke({"text": beispieltext})
structured_result = structured_chain.invoke({"text": beispieltext})

print("Ergebnis mit einfachem Prompt:")
print(basic_result)
print("\nErgebnis mit detailliertem Prompt:")
print(detailed_result)
print("\nErgebnis mit strukturiertem Prompt:")
print(structured_result)

# 4. Analysieren Sie die Unterschiede zwischen den Ergebnissen:
# - Welche Version liefert die präziseste Zusammenfassung?
# - Welche Version hält sich am besten an die Vorgabe von genau 3 Sätzen?
# - Was könnten Sie noch verbessern?
```

**Tipps:**
- Achten Sie auf die unterschiedliche Strukturierung der Prompts
- Notieren Sie, welche Elemente die Qualität der Antworten verbessern
- Experimentieren Sie mit weiteren Varianten, um den optimalen Prompt zu finden

## Übung 2: Prompt-Versionierung mit LangChain Hub

**Aufgabe:** Erstellen und verwalten Sie verschiedene Versionen eines Prompts im LangChain Hub.

```python
import os
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

# Initialisierung
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 1. Erstellen Sie einen eigenen Prompt für eine spezifische Aufgabe
# Beispiel: Ein Prompt zur Erstellung von Produktbeschreibungen
product_prompt_v1 = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein erfahrener Marketing-Texter, der überzeugende Produktbeschreibungen verfasst."),
    ("human", """
    Erstelle eine ansprechende Produktbeschreibung für folgendes Produkt:
    
    Produktname: {produktname}
    Kategorie: {kategorie}
    Hauptmerkmale: {merkmale}
    Zielgruppe: {zielgruppe}
    
    Die Beschreibung sollte 3-4 Sätze umfassen und die USPs hervorheben.
    """)
])

# 2. Testen Sie Ihren Prompt lokal
test_input = {
    "produktname": "EcoFresh 3000",
    "kategorie": "Wasserfilter",
    "merkmale": "Filtert 99,9% aller Schadstoffe, langlebiger Filter (6 Monate), kompaktes Design, umweltfreundliche Materialien",
    "zielgruppe": "Umweltbewusste Haushalte, die Wert auf sauberes Trinkwasser legen"
}

chain = product_prompt_v1 | llm | StrOutputParser()
result_v1 = chain.invoke(test_input)
print("Ergebnis mit Version 1:")
print(result_v1)

# 3. Erstellen Sie eine verbesserte Version Ihres Prompts
product_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", """Du bist ein spezialisierter Marketing-Experte, der überzeugende und verkaufsfördernde Produktbeschreibungen erstellt. 
    Deine Texte sind prägnant, nutzen emotionale Sprache und überzeugen durch Hervorhebung der Vorteile statt nur Funktionen."""),
    ("human", """
    Erstelle eine verkaufsfördernde Produktbeschreibung für:
    
    Produktname: {produktname}
    Kategorie: {kategorie}
    Hauptmerkmale: {merkmale}
    Zielgruppe: {zielgruppe}
    
    Folge diesem Format:
    1. Beginne mit einem emotionalen Aufhänger
    2. Beschreibe die wichtigsten 2-3 Vorteile (nicht Funktionen) des Produkts
    3. Schließe mit einem Handlungsaufruf ab
    
    Gesamtlänge: 4-5 Sätze
    Ton: Überzeugend, aber nicht übertrieben werblich
    """)
])

# 4. Testen der verbesserten Version
result_v2 = (product_prompt_v2 | llm | StrOutputParser()).invoke(test_input)
print("\nErgebnis mit Version 2:")
print(result_v2)

# 5. Optional: Prompt im LangChain Hub veröffentlichen
# Hinweis: Dafür benötigen Sie ein LangChain Hub Konto und API-Schlüssel

# Uncomment diese Zeilen, wenn Sie den Prompt im Hub speichern möchten
# Setzen Sie vorher Ihre LangChain Hub API-Schlüssel als Umgebungsvariable
# os.environ["LANGCHAIN_HUB_API_KEY"] = "Ihr API-Schlüssel hier"
# hub.push("IhrUsername/produktbeschreibung", product_prompt_v2)

# 6. Optional: Laden eines Prompts aus dem Hub
# Ein Beispiel für das Laden eines existierenden Prompts
# sentiment_prompt = hub.pull("borislove/customer-sentiment-analysis")
```

**Tipps:**
- Vergleichen Sie die Ergebnisse beider Prompt-Versionen
- Dokumentieren Sie klar, was in Version 2 verbessert wurde
- Verwenden Sie aussagekräftige Namen für Ihre Prompts im Hub

## Übung 3: Prompt-Management in Teams

**Aufgabe:** Erstellen Sie ein Prompt-Template-System für ein fiktives Team von KI-Entwicklern.

```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
import json

# Konzept: Ein zentrales Prompt-Repository für verschiedene Anwendungsfälle
# in einem Team mit mehreren Entwicklern

# 1. Erstellen Sie eine Basis-Prompt-Klasse für Ihr Team
class TeamPromptTemplate:
    """Basis-Klasse für alle Team-Prompts mit Metadaten und Versionierung"""
    
    def __init__(self, name, version, author, description, template):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.template = template
        
    def get_prompt(self):
        """Gibt das LangChain-Prompt-Objekt zurück"""
        return self.template
    
    def to_json(self):
        """Serialisiert die Prompt-Metadaten zu JSON"""
        metadata = {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            # Hinweis: Das eigentliche Template kann nicht einfach serialisiert werden
            # In einer echten Implementierung würde man hier die Template-Strings speichern
        }
        return json.dumps(metadata, indent=2)
    
    def __str__(self):
        return f"{self.name} v{self.version} by {self.author}"

# 2. Erstellen Sie einige Team-Prompts für verschiedene Anwendungsfälle

# a) Prompt für Datenanalyse
data_analysis_prompt = TeamPromptTemplate(
    name="DataAnalysisHelper",
    version="1.0",
    author="data_team",
    description="Hilft bei der Analyse und Interpretation von Datensätzen",
    template=ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Datenanalyse-Experte, der komplexe Datensätze interpretiert und klare Erkenntnisse liefert."),
        ("human", """
        Analysiere die folgenden Daten und gib eine klare Interpretation:
        
        Datensatz: {data}
        
        Bitte berücksichtige:
        1. Haupttrends und Muster
        2. Auffällige Ausreißer oder Anomalien
        3. Mögliche Schlussfolgerungen
        4. Empfehlungen für weitere Analysen
        
        Format deine Antwort übersichtlich mit Zwischenüberschriften.
        """)
    ])
)

# b) Prompt für Kundensupport
customer_support_prompt = TeamPromptTemplate(
    name="CustomerSupportAssistant",
    version="1.2",
    author="support_team",
    description="Generiert freundliche und hilfsbereite Antworten auf Kundenanfragen",
    template=ChatPromptTemplate.from_messages([
        ("system", """Du bist ein freundlicher Kundensupport-Mitarbeiter. 
        Du kommunizierst höflich, lösungsorientiert und empathisch.
        Du vermeidest technischen Jargon und erklärst komplexe Sachverhalte einfach."""),
        ("human", """
        Beantworte die folgende Kundenanfrage:
        
        Kundenanfrage: {customer_query}
        Produktkategorie: {product_category}
        Kundenstatus: {customer_status}
        
        Stelle sicher, dass deine Antwort:
        - Höflich und verständnisvoll ist
        - Eine klare Lösung oder nächste Schritte anbietet
        - Bei Premium-Kunden zusätzliche Optionen erwähnt
        """)
    ])
)

# c) Prompt für Content-Erstellung
content_creation_prompt = TeamPromptTemplate(
    name="ContentGenerator",
    version="2.1",
    author="marketing_team",
    description="Erstellt verschiedene Arten von Marketing-Content",
    template=ChatPromptTemplate.from_messages([
        ("system", "Du bist ein kreativer Content-Creator mit Expertise in digitalen Marketing-Formaten."),
        ("human", """
        Erstelle {content_type} zum Thema: {topic}
        
        Stilrichtung: {style}
        Zielgruppe: {target_audience}
        Länge: {length}
        Kernbotschaft: {key_message}
        Call-to-Action: {cta}
        
        Der Content sollte die Markenidentität widerspiegeln und SEO-optimiert sein.
        """)
    ])
)

# 3. Testen Sie die verschiedenen Team-Prompts

# Initialisierung des Modells
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Test des Datenanalyse-Prompts
data_example = """
Jahr, Umsatz, Kosten, Gewinn
2020, 1200000, 800000, 400000
2021, 1500000, 950000, 550000
2022, 1350000, 1050000, 300000
2023, 1800000, 1100000, 700000
"""

data_analysis_chain = data_analysis_prompt.get_prompt() | llm | output_parser
data_analysis_result = data_analysis_chain.invoke({"data": data_example})

print(f"Test von {data_analysis_prompt}:")
print(data_analysis_result)
print("\n" + "-"*50 + "\n")

# Test des Kundensupport-Prompts
customer_query_example = {
    "customer_query": "Mein neuer EcoFresh 3000 Filter macht seltsame Geräusche beim Filtern. Ist das normal oder ist er defekt?",
    "product_category": "Wasserfilter",
    "customer_status": "Premium"
}

customer_support_chain = customer_support_prompt.get_prompt() | llm | output_parser
customer_support_result = customer_support_chain.invoke(customer_query_example)

print(f"Test von {customer_support_prompt}:")
print(customer_support_result)
print("\n" + "-"*50 + "\n")

# 4. Prompt-Metadaten und Versionsverwaltung demonstrieren
print("Prompt-Metadaten:")
print(data_analysis_prompt.to_json())
print(customer_support_prompt.to_json())
print(content_creation_prompt.to_json())

# 5. Fragen zur Reflektion:
# - Wie könnten Sie die Versionskontrolle in einer Produktionsumgebung implementieren?
# - Welche Rollen und Berechtigungen würden Sie für das Prompt-Management einrichten?
# - Wie würden Sie die Leistung verschiedener Prompt-Versionen messen und vergleichen?
```

**Tipps:**
- Überlegen Sie, wie Prompts in einer echten Team-Umgebung verwaltet werden könnten
- Denken Sie an die Skalierbarkeit und Wartbarkeit Ihres Prompt-Management-Systems
- Berücksichtigen Sie Performance-Metriken für verschiedene Prompt-Versionen

## Übung 4: Integration mit Langfuse für Prompt-Tracking

**Aufgabe:** Integrieren Sie Langfuse zur Überwachung und Analyse verschiedener Prompt-Versionen.

```python
# Hinweis: Um diese Übung durchzuführen, benötigen Sie:
# - Ein Langfuse-Konto (kostenlos verfügbar)
# - Die Langfuse API-Schlüssel als Umgebungsvariablen
# pip install langfuse langchain-langfuse

import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langfuse.client import Langfuse
from langfuse.model import CreateTrace
from datetime import datetime

# 1. Langfuse Client einrichten (Setzen Sie Ihre Umgebungsvariablen)
# Uncomment die folgenden Zeilen und setzen Sie Ihre Schlüssel, wenn Sie Langfuse verwenden
# os.environ["LANGFUSE_PUBLIC_KEY"] = "your-public-key"
# os.environ["LANGFUSE_SECRET_KEY"] = "your-secret-key"
# os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # oder Ihre eigene Instanz

# Für diese Übung simulieren wir Langfuse-Tracking ohne tatsächliche API-Aufrufe
class MockLangfuse:
    def __init__(self):
        self.traces = []
        print("Mock Langfuse initialized (for demonstration)")
        
    def trace(self, name, **kwargs):
        trace_id = f"trace_{len(self.traces) + 1}"
        self.traces.append({
            "id": trace_id,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "metadata": kwargs
        })
        print(f"Created trace: {name} (ID: {trace_id})")
        return MockTrace(trace_id, self)

class MockTrace:
    def __init__(self, trace_id, client):
        self.trace_id = trace_id
        self.client = client
        self.spans = []
        
    def span(self, name, **kwargs):
        span_id = f"span_{len(self.spans) + 1}"
        self.spans.append({
            "id": span_id,
            "name": name,
            "trace_id": self.trace_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": kwargs
        })
        print(f"  Created span: {name} (ID: {span_id})")
        return MockSpan(span_id, self)
    
    def update(self, **kwargs):
        print(f"  Updated trace {self.trace_id} with: {kwargs}")
        return self

class MockSpan:
    def __init__(self, span_id, trace):
        self.span_id = span_id
        self.trace = trace
        
    def end(self, **kwargs):
        print(f"    Ended span {self.span_id} with: {kwargs}")
        return self

# Langfuse Client initialisieren (mock oder real)
try:
    langfuse = Langfuse()
    print("Real Langfuse client initialized")
except:
    langfuse = MockLangfuse()

# 2. Verschiedene Prompt-Versionen definieren
product_description_prompts = {
    "v1_basic": ChatPromptTemplate.from_template(
        "Beschreibe das Produkt {product_name} in 2-3 Sätzen."
    ),
    "v2_detailed": ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Marketing-Experte."),
        ("human", "Beschreibe das Produkt {product_name} mit seinen Hauptmerkmalen in 3-4 Sätzen.")
    ]),
    "v3_structured": ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Premium-Marketing-Experte mit Fokus auf überzeugenden Produktbeschreibungen."),
        ("human", """
        Erstelle eine Produktbeschreibung für: {product_name}
        
        Verwendungszweck: {purpose}
        Hauptmerkmale: {features}
        
        Format:
        1. Beginne mit einem Aufmerksamkeits-Satz
        2. Beschreibe die 2-3 wichtigsten Vorteile
        3. Schließe mit einem überzeugenden Fazit
        """)
    ])
}

# 3. Funktion zum Testen und Tracken von Prompt-Versionen
def test_prompt_version(version_name, prompt, inputs, llm):
    # Trace für diese Prompt-Version starten
    trace = langfuse.trace(
        name="prompt_test",
        metadata={
            "prompt_version": version_name,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    # Prompt-Span erstellen
    prompt_span = trace.span(
        name="prompt_preparation",
        metadata={"prompt_template": str(prompt)}
    )
    
    # Formatierter Prompt erstellen
    formatted_prompt = prompt.format_messages(**inputs)
    prompt_span.end(output={"formatted_prompt": str(formatted_prompt)})
    
    # LLM-Aufruf-Span
    llm_span = trace.span(name="llm_call", input=inputs)
    
    try:
        # Chain ausführen
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(inputs)
        
        # Erfolgreichen Aufruf tracken
        llm_span.end(output={"result": result})
        trace.update(status="success")
        
        return result
    except Exception as e:
        # Fehler tracken
        llm_span.end(status="error", error={"message": str(e)})
        trace.update(status="error", error={"message": str(e)})
        raise e

# 4. Verschiedene Prompt-Versionen testen
llm = ChatOpenAI(model="gpt-3.5-turbo")

test_inputs = {
    "simple": {
        "product_name": "EcoFresh 3000 Wasserfilter"
    },
    "detailed": {
        "product_name": "EcoFresh 3000 Wasserfilter",
        "purpose": "Reinigung von Leitungswasser",
        "features": "Filtert 99,9% aller Schadstoffe, langlebiger Filter (6 Monate), kompaktes Design"
    }
}

print("\n=== Testen verschiedener Prompt-Versionen mit Tracking ===\n")

# Einfache Version mit einfachen Inputs
print("Version 1 (Basic) Test:")
result_v1 = test_prompt_version("v1_basic", product_description_prompts["v1_basic"], 
                              test_inputs["simple"], llm)
print(f"Ergebnis:\n{result_v1}\n")

# Detailliertere Version mit einfachen Inputs
print("Version 2 (Detailed) Test:")
result_v2 = test_prompt_version("v2_detailed", product_description_prompts["v2_detailed"], 
                              test_inputs["simple"], llm)
print(f"Ergebnis:\n{result_v2}\n")

# Strukturierte Version mit detaillierten Inputs
print("Version 3 (Structured) Test:")
result_v3 = test_prompt_version("v3_structured", product_description_prompts["v3_structured"],
                              test_inputs["detailed"], llm)
print(f"Ergebnis:\n{result_v3}\n")

# 5. Auswertung 
print("\n=== Auswertung ===\n")
print("In einer echten Langfuse-Implementierung würden Sie nun im Dashboard:")
print("1. Die Performance verschiedener Prompt-Versionen vergleichen")
print("2. Antwortzeiten und Token-Verbrauch analysieren")
print("3. Erfolgsraten verschiedener Prompts überwachen")
print("4. A/B-Tests zwischen Prompt-Versionen durchführen")
print("5. Kostenoptimierung basierend auf Token-Nutzung vornehmen")

# 6. Fragen zur Reflektion:
# - Welche Metriken wären für die Bewertung von Prompts am wichtigsten?
# - Wie könnten Sie A/B-Tests zwischen verschiedenen Prompt-Versionen implementieren?
# - Wie würden Sie das Prompt-Management in einen CI/CD-Prozess integrieren?
```

**Tipps:**
- Überlegen Sie, welche Metriken für das Prompt-Tracking relevant sind
- Denken Sie an die Integration mit bestehenden DevOps- und CI/CD-Prozessen
- Berücksichtigen Sie auch qualitative Aspekte der Prompt-Evaluation

## Bonus-Übung: Automatische Prompt-Optimierung

**Aufgabe:** Implementieren Sie einen automatisierten Prozess zur Optimierung von Prompts basierend auf Feedback.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
import random

# Diese Übung zeigt, wie Prompts automatisch basierend auf Feedback optimiert werden können

# 1. Initialisierung
llm = ChatOpenAI(model="gpt-3.5-turbo")
optimizer_llm = ChatOpenAI(model="gpt-4")  # Für Prompt-Optimierung ein stärkeres Modell verwenden

# 2. Ausgangs-Prompt erstellen
initial_prompt = ChatPromptTemplate.from_template(
    "Erkläre das Konzept von {concept} einfach und verständlich."
)

# 3. Funktion zum Bewerten der Antworten
def rate_explanation(explanation, criteria):
    """Bewertet eine Erklärung anhand bestimmter Kriterien (simuliert)"""
    # In einer echten Anwendung würde hier ein LLM oder menschliches Feedback stehen
    # Für diese Übung simulieren wir die Bewertung
    
    words = explanation.split()
    
    scores = {
        "clarity": 0,        # Klarheit
        "completeness": 0,   # Vollständigkeit
        "conciseness": 0,    # Prägnanz
        "accuracy": 0        # Richtigkeit
    }
    
    # Sehr einfache simulierte Bewertung (in der Praxis würde hier ein LLM oder Mensch bewerten)
    scores["clarity"] = min(10, max(1, 10 - (random.randint(0, 5))))
    scores["completeness"] = min(10, max(1, len(words) / 30))
    scores["conciseness"] = min(10, max(1, 20 - (len(words) / 40)))
    scores["accuracy"] = min(10, max(1, 8 + random.randint(-2, 2)))
    
    # Gewichtete Gesamtbewertung basierend auf den angegebenen Kriterien
    total_score = 0
    total_weight = 0
    
    for criterion, weight in criteria.items():
        total_score += scores[criterion] * weight
        total_weight += weight
    
    final_score = total_score / total_weight
    
    return {
        "detailed_scores": scores,
        "final_score": final_score
    }

# 4. Funktion zum Optimieren des Prompts basierend auf Feedback
def optimize_prompt(prompt_template, concept, feedback, iteration):
    optimizer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Prompt-Engineering-Experte, der Prompts basierend auf Feedback verbessert."),
        ("human", f"""
        Ich habe den folgenden Prompt verwendet:
        
        ```
        {prompt_template}
        ```
        
        Beim Testen mit dem Konzept "{concept}" erhielt ich folgende Bewertungen:
        
        {feedback}
        
        Dies ist Iteration {iteration}/3. Bitte verbessere den Prompt, um bessere Ergebnisse zu erzielen.
        Achte besonders auf die Bereiche mit niedrigen Bewertungen.
        
        Liefere nur den verbesserten Prompt-Text ohne zusätzliche Erklärungen.
        """)
    ])
    
    new_prompt_text = (optimizer_prompt | optimizer_llm | StrOutputParser()).invoke({})
    
    # Neuen Prompt erstellen
    return ChatPromptTemplate.from_template(new_prompt_text)

# 5. Hauptfunktion zum Ausführen des Optimierungsprozesses
def run_prompt_optimization(initial_prompt, concept, iterations=3):
    current_prompt = initial_prompt
    
    print(f"Optimierung für Konzept: {concept}")
    print(f"Ausgangsprompt: {initial_prompt.template}\n")
    
    # Bewertungskriterien definieren
    criteria = {
        "clarity": 1.0,       # Klarheit ist sehr wichtig
        "completeness": 0.8,  # Vollständigkeit ist wichtig
        "conciseness": 0.6,   # Prägnanz ist etwas weniger wichtig
        "accuracy": 1.0       # Richtigkeit ist sehr wichtig
    }
    
    best_score = 0
    best_prompt = current_prompt
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}:")
        print(f"Aktueller Prompt: {current_prompt.template}")
        
        # Prompt anwenden
        chain = current_prompt | llm | StrOutputParser()
        explanation = chain.invoke({"concept": concept})
        
        print(f"\nErklärung: {explanation[:150]}...\n")
        
        # Erklärung bewerten
        scores = rate_explanation(explanation, criteria)
        print(f"Bewertungen: {scores['detailed_scores']}")
        print(f"Gesamtbewertung: {scores['final_score']:.2f}/10\n")
        
        # Besten Prompt speichern
        if scores['final_score'] > best_score:
            best_score = scores['final_score']
            best_prompt = current_prompt
        
        # Wenn nicht die letzte Iteration, dann Prompt optimieren
        if i < iterations - 1:
            feedback_str = f"""
            Detaillierte Bewertungen:
            - Klarheit: {scores['detailed_scores']['clarity']}/10
            - Vollständigkeit: {scores['detailed_scores']['completeness']}/10
            - Prägnanz: {scores['detailed_scores']['conciseness']}/10
            - Richtigkeit: {scores['detailed_scores']['accuracy']}/10
            
            Gesamtbewertung: {scores['final_score']:.2f}/10
            """
            
            current_prompt = optimize_prompt(current_prompt.template, concept, feedback_str, i+1)
            print("-" * 50)
    
    print("\n" + "=" * 50)
    print(f"Optimierung abgeschlossen!")
    print(f"Bester Prompt (Bewertung: {best_score:.2f}/10):")
    print(best_prompt.template)
    print("=" * 50)
    
    return best_prompt

# 6. Optimierungsprozess für verschiedene Konzepte ausführen
test_concepts = ["Vektordatenbanken", "Prompt-Engineering", "Fine-Tuning von LLMs"]

for concept in test_concepts:
    optimized_prompt = run_prompt_optimization(initial_prompt, concept, iterations=3)
    print("\n")

# 7. Fragen zur Reflektion:
# - Wie könnten Sie die Bewertungsfunktion in einer realen Anwendung implementieren?
# - Welche anderen Aspekte könnten bei der Prompt-Optimierung berücksichtigt werden?
# - Wie könnte man diesen Prozess in eine kontinuierliche Verbesserungsschleife integrieren?
```

Ich fahre mit dem Code für die Bonus-Übung "Automatische Prompt-Optimierung" fort:

```python workshop/03_03_prompt_management_uebungen.md
        # Prompt anwenden
        chain = current_prompt | llm | StrOutputParser()
        explanation = chain.invoke({"concept": concept})
        
        print(f"\nErklärung: {explanation[:150]}...\n")
        
        # Erklärung bewerten
        scores = rate_explanation(explanation, criteria)
        print(f"Bewertungen: {scores['detailed_scores']}")
        print(f"Gesamtbewertung: {scores['final_score']:.2f}/10\n")
        
        # Besten Prompt speichern
        if scores['final_score'] > best_score:
            best_score = scores['final_score']
            best_prompt = current_prompt
        
        # Wenn nicht die letzte Iteration, dann Prompt optimieren
        if i < iterations - 1:
            feedback_str = f"""
            Detaillierte Bewertungen:
            - Klarheit: {scores['detailed_scores']['clarity']}/10
            - Vollständigkeit: {scores['detailed_scores']['completeness']}/10
            - Prägnanz: {scores['detailed_scores']['conciseness']}/10
            - Richtigkeit: {scores['detailed_scores']['accuracy']}/10
            
            Gesamtbewertung: {scores['final_score']:.2f}/10
            """
            
            current_prompt = optimize_prompt(current_prompt.template, concept, feedback_str, i+1)
            print("-" * 50)
    
    print("\n" + "=" * 50)
    print(f"Optimierung abgeschlossen!")
    print(f"Bester Prompt (Bewertung: {best_score:.2f}/10):")
    print(best_prompt.template)
    print("=" * 50)
    
    return best_prompt

# 6. Optimierungsprozess für verschiedene Konzepte ausführen
test_concepts = ["Vektordatenbanken", "Prompt-Engineering", "Fine-Tuning von LLMs"]

for concept in test_concepts:
    optimized_prompt = run_prompt_optimization(initial_prompt, concept, iterations=3)
    print("\n")

# 7. Fragen zur Reflektion:
# - Wie könnten Sie die Bewertungsfunktion in einer realen Anwendung implementieren?
# - Welche anderen Aspekte könnten bei der Prompt-Optimierung berücksichtigt werden?
# - Wie könnte man diesen Prozess in eine kontinuierliche Verbesserungsschleife integrieren?
```

**Tipps:**
- In einer realen Anwendung würden Sie die Bewertung durch ein LLM oder durch echtes Nutzer-Feedback ersetzen
- Achten Sie auf die Balance zwischen verschiedenen Optimierungskriterien
- Denken Sie daran, dass der Prompt-Optimierungsprozess selbst Kosten verursacht (API-Aufrufe)

## Zusammenfassung der Übungen

Die Übungen in diesem Modul haben folgende Themen abgedeckt:

1. **Strukturierte Prompts erstellen und optimieren**: Vergleich verschiedener Prompt-Strukturen und deren Einfluss auf die Ergebnisqualität 
2. **Prompt-Versionierung mit LangChain Hub**: Praktische Anwendung der Prompt-Versionierung und -Verwaltung im Team
3. **Prompt-Management in Teams**: Entwicklung eines Template-Systems für die strukturierte Verwaltung von Prompts in Teamumgebungen
4. **Integration mit Langfuse für Prompt-Tracking**: Überwachung und Analyse von Prompt-Leistung für kontinuierliche Verbesserung
5. **Automatische Prompt-Optimierung**: Implementierung eines Systems zur algorithmischen Verbesserung von Prompts basierend auf Feedback

Diese Übungen helfen Ihnen, effektivere Prompts zu entwickeln und einen strukturierten Prozess für das Prompt-Management in größeren Projekten zu etablieren.
