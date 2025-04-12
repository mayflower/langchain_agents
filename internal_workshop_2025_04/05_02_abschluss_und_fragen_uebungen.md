# Abschluss und Fragen - Übungen

## Übung 1: LLM-Anwendungsdesign

**Aufgabe:** Entwerfen Sie eine LLM-Anwendung, die mindestens drei im Workshop behandelte Konzepte kombiniert.

1. Wählen Sie ein Anwendungsszenario aus einem der folgenden Bereiche:
   - Kundenservice-Automation
   - Dokumentenanalyse und -zusammenfassung
   - Persönlicher Assistent für Recherche
   - Datenanalyse und Reporting
   - Content-Erstellung

2. Beschreiben Sie die Anforderungen Ihrer Anwendung:
   - Welches Problem löst sie?
   - Wer sind die Nutzer?
   - Welche Datenquellen werden benötigt?

3. Entwerfen Sie eine Architektur mit mindestens drei der folgenden Komponenten:
   - RAG-System mit Vektordatenbank
   - Tool-Calling für externe Funktionalitäten
   - LangGraph für Workflow-Management
   - Memory-Komponente für Zustandsverwaltung
   - Strukturierte Ausgabe für spezifische Datenformate
   - Evaluationskomponente für Qualitätssicherung

4. Erstellen Sie ein einfaches Diagramm Ihrer Architektur (kann von Hand gezeichnet sein)

5. Schreiben Sie Pseudocode für ein oder zwei Kernkomponenten Ihrer Anwendung

**Hinweise:**
- Denken Sie an die Vor- und Nachteile verschiedener Ansätze, die wir diskutiert haben
- Berücksichtigen Sie Aspekte wie Skalierbarkeit, Kosten und Datenschutz
- Sie können existierende Code-Beispiele aus dem Workshop als Ausgangspunkt verwenden

## Übung 2: Workshop-Rückblick und Konsolidierung

**Aufgabe:** Erstellen Sie eine persönliche Zusammenfassung der wichtigsten Workshop-Erkenntnisse und planen Sie Ihre nächsten Lernschritte.

1. Schreiben Sie für jeden Workshop-Teil die für Sie wichtigsten Erkenntnisse auf (1-2 Sätze pro Teil):
   - Teil 1: Grundlagen
   - Teil 2: Architektur & State Management
   - Teil 3: Datenverwaltung & Infrastruktur
   - Teil 4: Tools & Frameworks
   - Teil 5: Qualitätssicherung & Evaluierung

2. Identifizieren Sie:
   - 3 Konzepte, die Sie sofort in eigenen Projekten anwenden können
   - 2 Themen, zu denen Sie mehr lernen möchten
   - 1 konkretes nächstes Projekt, bei dem Sie das Gelernte anwenden wollen

3. Erstellen Sie einen persönlichen "Lernfahrplan" für die nächsten 4 Wochen:
   - Woche 1: Welche Ressourcen werden Sie nutzen, um Ihr Wissen zu vertiefen?
   - Woche 2: Welche kleinen Experimente oder Proof-of-Concepts planen Sie?
   - Woche 3-4: Wie werden Sie Ihr konkretes Projekt umsetzen?

4. Definieren Sie 3-5 spezifische Fragen, die nach dem Workshop noch offen geblieben sind

**Hinweise:**
- Dies ist eine persönliche Übung, es gibt keine falschen Antworten
- Der Fokus liegt auf der Anwendbarkeit in Ihrem beruflichen oder persönlichen Kontext
- Die Übung hilft Ihnen, das Gelernte zu verankern und konkrete nächste Schritte zu planen

## Übung 3: Mini-Hackathon - Implementierung einer RAG-Anwendung

**Aufgabe:** Implementieren Sie eine einfache RAG-Anwendung, die verschiedene Workshop-Inhalte zusammenführt.

1. Erstellen Sie eine einfache RAG-Anwendung mit folgenden Komponenten:
   - Verwendung einer beliebigen Textsammlung als Wissensbasis (z.B. Workshop-Materialien)
   - Aufteilung der Dokumente in Chunks
   - Erstellung und Speicherung von Embeddings
   - Retrieval relevanter Dokumente zu einer Anfrage
   - Generierung einer Antwort unter Verwendung des abgerufenen Kontexts
   - Einfache Evaluierung der Antwortqualität

2. Implementieren Sie folgendes Beispiel oder entwickeln Sie es weiter:

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.evaluation import load_evaluator

# Umgebungsvariablen laden
load_dotenv()

# 1. Dokumente laden
# - Passen Sie den Pfad zu Ihrer Textsammlung an
loader = DirectoryLoader("./workshop_materials", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# 2. Dokumente aufteilen
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Embeddings erstellen und in Vektordatenbank speichern
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. RAG-Prompt definieren
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Du bist ein hilfreicher Assistent, der Fragen basierend auf dem bereitgestellten Kontext beantwortet.
    Verwende nur die Informationen aus dem Kontext, um die Frage zu beantworten.
    Wenn du die Antwort nicht im Kontext findest, sage ehrlich, dass du es nicht weißt.
    
    Kontext:
    {context}
    """),
    ("human", "{question}")
])

# 5. LLM definieren
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 6. RAG-Chain erstellen
def retrieve(query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_response(query):
    context = retrieve(query)
    chain = rag_prompt | llm | StrOutputParser()
    return {"response": chain.invoke({"context": context, "question": query}), 
            "context": context}

# 7. Einfacher Evaluator
evaluator = load_evaluator("qa")

# 8. Testfunktion
def test_rag_system(query):
    result = generate_response(query)
    response = result["response"]
    context = result["context"]
    
    print(f"Frage: {query}\n")
    print(f"Antwort: {response}\n")
    
    # Einfache Evaluation
    eval_result = evaluator.evaluate_strings(
        prediction=response,
        input=query,
        reference=context  # Als "Gold-Standard" verwenden wir hier den Kontext
    )
    
    print(f"Evaluation: {eval_result}")
    return result

# Beispielanfrage testen
test_query = "Was ist LangGraph und wie funktioniert es?"
test_rag_system(test_query)
```

3. Erweitern Sie das Beispiel um mindestens zwei der folgenden Funktionen:
   - Integration von LangFuse für Monitoring und Tracing
   - Hinzufügen eines Tools zur Web-Suche für aktuelle Informationen
   - Implementierung eines LangGraph-basierten Workflows für mehrstufige Antwortgenerierung
   - Speicherung der Konversationshistorie mit einem Memory-Typ
   - Implementierung eines A/B-Tests mit zwei verschiedenen Prompt-Varianten

**Hinweise:**
- Achten Sie auf den Tokenverbrauch und testen Sie mit kleinen Dokumentensammlungen
- Sie können Markdown-Dateien aus dem Workshop als Testdaten verwenden
- Diese Übung kann in Kleingruppen durchgeführt werden
- Dokumentieren Sie Ihre Ergebnisse und Erkenntnisse aus der Implementierung

## Übung 4: Evaluierung Ihrer Workshop-Erkenntnisse

**Aufgabe:** Entwickeln Sie ein eigenes LLM-as-a-judge Evaluierungssystem für einen spezifischen Anwendungsfall.

1. Wählen Sie einen Anwendungsfall, für den Sie eine LLM-Anwendung entwickeln möchten (oder bereits entwickelt haben)

2. Definieren Sie 3-5 spezifische Evaluierungskriterien, die für diesen Anwendungsfall besonders wichtig sind, z.B.:
   - Faktische Korrektheit
   - Relevanz zur Anfrage
   - Vollständigkeit der Antwort
   - Konsistenz über mehrere Anfragen
   - Benutzerfreundlichkeit/Verständlichkeit
   - Domain-spezifische Kriterien

3. Erstellen Sie einen Prompt für die LLM-as-a-judge Evaluierung, der Ihre definierten Kriterien berücksichtigt

4. Implementieren Sie eine einfache Evaluierungsfunktion, die:
   - Eine Anfrage und die generierte Antwort annimmt
   - Ein Evaluierungsmodell verwendet, um die Antwort anhand Ihrer Kriterien zu bewerten
   - Einen strukturierten Bericht zurückgibt (z.B. als JSON)

5. Testen Sie Ihre Evaluierungsfunktion mit mindestens drei verschiedenen Anfrage-Antwort-Paaren

**Hinweise:**
- Verwenden Sie das im Workshop besprochene Format für LLM-as-a-judge Evaluierungen
- Sie können für die Evaluierung ein anderes (idealerweise stärkeres) Modell verwenden als für die Antwortgenerierung
- Denken Sie über Möglichkeiten nach, wie Sie die Ergebnisse quantifizieren können

**Beispiel für einen Evaluierungs-Prompt:**

```
Du bist ein Experte für die Evaluierung von KI-generierten Antworten in [Ihrem Domain-Bereich].
Bewerte die folgende Antwort auf eine Nutzeranfrage anhand dieser Kriterien:

1. Faktische Korrektheit (0-10): Sind alle Informationen in der Antwort sachlich korrekt?
2. Relevanz (0-10): Wie gut beantwortet die Antwort die ursprüngliche Anfrage?
3. Vollständigkeit (0-10): Deckt die Antwort alle wichtigen Aspekte der Anfrage ab?
4. [Weiteres spezifisches Kriterium] (0-10): [Beschreibung des Kriteriums]

Anfrage: {query}
Antwort: {response}

Gib für jedes Kriterium eine Punktzahl und eine kurze Begründung. 
Fasse dann die Gesamtbewertung zusammen und gib konkrete Verbesserungsvorschläge.
Formatiere deine Bewertung als JSON mit den Feldern: scores, explanations, overall_score, improvement_suggestions.
```

## Bonus-Übung: Entwicklung eines Agententeams

**Aufgabe:** Entwerfen und implementieren Sie ein Team von spezialisierten Agenten, die zusammenarbeiten, um komplexe Aufgaben zu lösen.

1. Definieren Sie ein komplexes Problem, das in mehrere Teilprobleme zerlegt werden kann, z.B.:
   - Recherche, Analyse und Zusammenfassung zu einem Thema
   - Datenanalyse, Visualisierung und Berichterstellung
   - Produktentwicklung mit Marktanalyse, Ideenfindung und Prototyping

2. Gestalten Sie 3-4 spezialisierte Agenten mit unterschiedlichen Rollen:
   - Welche Fähigkeiten hat jeder Agent?
   - Welche Tools stehen jedem Agenten zur Verfügung?
   - Wie kommunizieren die Agenten miteinander?

3. Implementieren Sie einen Orchestrator-Agenten, der:
   - Die Aufgabe in Teilaufgaben zerlegt
   - Die spezialisierten Agenten koordiniert
   - Die Ergebnisse zusammenführt und eine kohärente Ausgabe erzeugt

4. Testen Sie Ihr Agententeam mit mindestens einem komplexen Beispielszenario

**Hinweise:**
- Sie können die im Workshop vorgestellten LangGraph-Konzepte für die Koordination verwenden
- Experimentieren Sie mit verschiedenen Kommunikationsstrukturen (hierarchisch, Peer-to-Peer, etc.)
- Diese Übung ist anspruchsvoll und kann als längerfristiges Projekt nach dem Workshop fortgeführt werden
