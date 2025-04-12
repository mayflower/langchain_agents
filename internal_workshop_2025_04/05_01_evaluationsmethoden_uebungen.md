# Evaluationsmethoden - Übungen

## Übung 1: LLM-As-A-Judge implementieren

**Aufgabe:** Implementieren Sie einen LLM-As-A-Judge Evaluator, der zwei verschiedene Antworten auf die gleiche Frage vergleicht.

1. Erstellen Sie eine Funktion `evaluate_responses`, die zwei Antworten auf die gleiche Frage mit einem LLM bewertet
2. Die Funktion soll folgende Parameter akzeptieren:
   - `question`: Die ursprüngliche Frage
   - `answer_a`: Die erste Antwort
   - `answer_b`: Die zweite Antwort
   - `criteria`: Eine Liste von Bewertungskriterien (z.B. ["Korrektheit", "Vollständigkeit", "Klarheit"])
3. Die Funktion soll einen strukturierten Output zurückgeben, der für jedes Kriterium eine Bewertung sowie eine Gesamtempfehlung enthält

**Beispiel:**
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

def evaluate_responses(question, answer_a, answer_b, criteria):
    # Implementieren Sie hier die Bewertungsfunktion
    # Verwenden Sie einen strukturierten Output-Parser für einheitliche Ergebnisse
    
    # Beispiel für einen Rückgabewert:
    # {
    #   "criteria_scores": {
    #     "Korrektheit": {"answer_a": 8, "answer_b": 9, "reasoning": "..."},
    #     "Vollständigkeit": {"answer_a": 7, "answer_b": 6, "reasoning": "..."},
    #     "Klarheit": {"answer_a": 9, "answer_b": 7, "reasoning": "..."}
    #   },
    #   "overall_winner": "answer_a",
    #   "reasoning": "..."
    # }
    pass

# Testen Sie die Funktion
question = "Was sind die Vorteile von Vektordatenbanken für RAG-Anwendungen?"
answer_a = """Vektordatenbanken bieten drei Hauptvorteile für RAG: Erstens ermöglichen sie semantische Suche durch Ähnlichkeitsvergleiche von Embeddings. Zweitens skalieren sie effizient mit wachsenden Datenmengen. Drittens unterstützen sie Metadatenfilterung zur Eingrenzung von Suchergebnissen."""
answer_b = """Vektordatenbanken sind wichtig für RAG-Anwendungen. Sie speichern Vektoren und können schnell ähnliche Vektoren finden. Das ist nützlich, wenn man viele Dokumente hat."""

result = evaluate_responses(
    question=question,
    answer_a=answer_a,
    answer_b=answer_b,
    criteria=["Korrektheit", "Vollständigkeit", "Klarheit", "Detailgrad"]
)
```

**Tipps:**
- Verwenden Sie einen Output-Parser für strukturierte Ergebnisse
- Formulieren Sie einen klaren Prompt, der objektive Bewertungskriterien definiert
- Fügen Sie einen Reasoning-Schritt hinzu, damit das LLM seine Entscheidung begründet

## Übung 2: ROUGE und BLEU Metriken anwenden

**Aufgabe:** Implementieren Sie eine Funktion, die die ROUGE und BLEU Scores für generierte Antworten im Vergleich zu Referenzantworten berechnet.

1. Installieren Sie die benötigten Bibliotheken (`rouge_score`, `nltk`)
2. Erstellen Sie eine Funktion `calculate_nlp_metrics`, die folgende Parameter akzeptiert:
   - `reference`: Referenztext (Gold-Standard)
   - `candidate`: Zu bewertender Text
3. Die Funktion soll folgende Metriken zurückgeben:
   - ROUGE-1, ROUGE-2 und ROUGE-L F1-Scores
   - BLEU Score
4. Testen Sie die Funktion mit verschiedenen Beispieltexten

**Beispiel:**
```python
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')

def calculate_nlp_metrics(reference, candidate):
    # Implementieren Sie hier die Metrikberechnung
    # Berechnen Sie sowohl ROUGE als auch BLEU Scores
    pass

# Testen Sie die Funktion mit verschiedenen Beispielen
reference_1 = "Vektordatenbanken sind spezialisierte Datenbanksysteme, die für die effiziente Speicherung und das Abrufen von hochdimensionalen Vektoren optimiert sind."
candidate_1 = "Vektordatenbanken sind Datenbanken, die für das Speichern und Abfragen von Vektoren entwickelt wurden."

reference_2 = "Machine Learning ist ein Teilbereich der künstlichen Intelligenz, der Systeme befähigt, aus Erfahrungen zu lernen und sich zu verbessern, ohne explizit programmiert zu werden."
candidate_2 = "Maschinelles Lernen ist ein KI-Bereich, bei dem Systeme automatisch aus Daten lernen und ihre Leistung verbessern können."

metrics_1 = calculate_nlp_metrics(reference_1, candidate_1)
metrics_2 = calculate_nlp_metrics(reference_2, candidate_2)
```

**Tipps:**
- ROUGE misst die Überlappung von N-Grammen zwischen Referenz und Kandidat
- BLEU konzentriert sich auf die Präzision (wie viele N-Gramme im Kandidaten sind auch in der Referenz enthalten)
- Achten Sie auf die Tokenisierung der Texte (Groß-/Kleinschreibung, Interpunktion)

## Übung 3: PII-Erkennung und Anonymisierung mit Microsoft Presidio

**Aufgabe:** Implementieren Sie eine Funktion zur Anonymisierung von personenbezogenen Daten in Texten mit Microsoft Presidio.

1. Installieren Sie Presidio (`pip install presidio-analyzer presidio-anonymizer`)
2. Erstellen Sie eine Funktion `anonymize_text`, die einen Text erhält und alle erkannten PII anonymisiert
3. Die Funktion soll folgende PII-Typen erkennen und anonymisieren:
   - Namen
   - E-Mail-Adressen
   - Telefonnummern
   - Adressen
   - Kreditkartennummern
4. Testen Sie die Funktion mit verschiedenen Beispieltexten, die PII enthalten

**Beispiel:**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def anonymize_text(text, language="de"):
    # Implementieren Sie hier die PII-Erkennung und Anonymisierung
    pass

# Testen Sie die Funktion
text_with_pii = """
Sehr geehrter Herr Dr. Max Mustermann,

Vielen Dank für Ihre Anfrage. Sie erreichen mich telefonisch unter +49 123 456789 oder per E-Mail an max.mustermann@example.com.

Ihre Kundennummer ist DE123456789 und Ihre Kreditkartendaten (VISA 4111 1111 1111 1111, gültig bis 12/25) haben wir erfasst.

Ihre Adresse:
Musterstraße 123
10115 Berlin
Deutschland

Mit freundlichen Grüßen,
Dr. Anna Schmidt
"""

anonymized_text = anonymize_text(text_with_pii)
print(anonymized_text)
```

**Tipps:**
- Stellen Sie sicher, dass Sie die richtigen Analysatoren für die deutsche Sprache konfigurieren
- Experimetieren Sie mit verschiedenen Anonymisierungsmethoden (z.B. Ersetzen durch Platzhalter, Maskieren)
- Achten Sie besonders auf falsch-positive und falsch-negative Erkennungen

## Übung 4: RAG-Evaluation implementieren

**Aufgabe:** Implementieren Sie ein umfassendes Evaluationssystem für ein RAG-System, das verschiedene Aspekte des Systems bewertet.

1. Erstellen Sie eine Funktion `evaluate_rag`, die folgende Parameter akzeptiert:
   - `query`: Die Suchanfrage
   - `retrieved_documents`: Die abgerufenen Dokumente
   - `generated_answer`: Die generierte Antwort
   - `reference_answer` (optional): Eine Referenzantwort (falls verfügbar)
2. Die Funktion soll folgende Aspekte bewerten:
   - Kontext-Relevanz: Wie relevant sind die abgerufenen Dokumente für die Anfrage?
   - Faithfulness (Treue): Wird die Antwort durch die abgerufenen Dokumente gestützt?
   - Antwort-Qualität: Wie gut beantwortet die generierte Antwort die Anfrage?
3. Die Funktion soll einen detaillierten Evaluationsbericht zurückgeben

**Beispiel:**
```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

def evaluate_rag(query, retrieved_documents, generated_answer, reference_answer=None):
    # Implementieren Sie hier die RAG-Evaluation
    # Bewerten Sie mindestens Kontext-Relevanz, Faithfulness und Antwort-Qualität
    pass

# Testen Sie die Funktion
query = "Welche Architekturmuster gibt es für RAG-Systeme?"
retrieved_docs = [
    "Dokument 1: RAG-Systeme können verschiedene Architekturmuster verwenden, darunter einfaches RAG, Multi-Query RAG und Hierarchisches RAG...",
    "Dokument 2: Ein wichtiger Aspekt von RAG ist die Chunking-Strategie, die bestimmt, wie Dokumente in kleinere Einheiten aufgeteilt werden...",
    "Dokument 3: Die Wahl der Vektordatenbank kann die Effizienz und Skalierbarkeit eines RAG-Systems erheblich beeinflussen..."
]
generated_answer = "Es gibt verschiedene RAG-Architekturmuster, darunter Basic RAG, Multi-Query RAG, Hierarchical RAG und Fusion RAG. Basic RAG ist der einfachste Ansatz, bei dem Dokumente direkt aus einer Vektordatenbank abgerufen werden. Multi-Query RAG erzeugt mehrere Variationen der ursprünglichen Anfrage, um die Abdeckung zu verbessern. Hierarchical RAG strukturiert den Abruf in mehreren Ebenen, während Fusion RAG Ergebnisse aus verschiedenen Retrieval-Methoden kombiniert."
reference_answer = "Die wichtigsten RAG-Architekturmuster sind Basic RAG, Multi-Query RAG, Hierarchical RAG und Hypothetical Document Embeddings (HyDE). Jedes Muster hat spezifische Vor- und Nachteile hinsichtlich Präzision, Rechenaufwand und Implementierungskomplexität."

evaluation_report = evaluate_rag(
    query=query,
    retrieved_documents=retrieved_docs,
    generated_answer=generated_answer,
    reference_answer=reference_answer
)
```

**Tipps:**
- Verwenden Sie LLM-basierte Bewertung für qualitative Aspekte wie Relevanz und Faithfulness
- Kombinieren Sie automatische Metriken (ROUGE, BLEU) mit LLM-Bewertungen für ein umfassendes Bild
- Experimentieren Sie mit verschiedenen Prompts für die LLM-basierte Bewertung

## Übung 5: Systematische Testdatensatz-Erstellung

**Aufgabe:** Entwickeln Sie einen Prozess zur systematischen Erstellung eines Testdatensatzes für die Evaluation Ihres RAG-Systems.

1. Erstellen Sie eine Funktion `generate_test_cases`, die einen bestehenden Corpus von Dokumenten nutzt, um automatisch Testfälle zu generieren
2. Jeder Testfall sollte folgende Elemente enthalten:
   - Eine Frage/Anfrage
   - Die erwarteten relevanten Dokumente
   - Eine Referenzantwort
3. Die Funktion sollte verschiedene Fragetypen abdecken:
   - Faktenfragen (z.B. "Wer hat...?", "Wann wurde...?")
   - Vergleichsfragen (z.B. "Was ist der Unterschied zwischen...?")
   - Ursachenfragen (z.B. "Warum passiert...?")
   - Methodenfragen (z.B. "Wie funktioniert...?")
4. Implementieren Sie auch eine Funktion `evaluate_test_cases`, die das RAG-System auf allen Testfällen evaluiert und Statistiken liefert

**Beispiel:**
```python
def generate_test_cases(document_corpus, num_test_cases=10):
    # Implementieren Sie hier die Testfall-Generierung
    # Verwenden Sie das LLM, um sinnvolle Fragen zu den Dokumenten zu generieren
    pass

def evaluate_test_cases(rag_system, test_cases):
    # Implementieren Sie hier die systematische Evaluation über alle Testfälle
    # Sammeln Sie Statistiken zu den verschiedenen Bewertungskriterien
    pass

# Testen Sie die Funktionen
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Beispiel-Dokumente laden (anpassen an Ihren Anwendungsfall)
loader = TextLoader("beispieldokumente.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

# Testfälle generieren
test_cases = generate_test_cases(split_docs, num_test_cases=5)

# RAG-System evaluieren
evaluation_results = evaluate_test_cases(my_rag_system, test_cases)
```

**Tipps:**
- Verwenden Sie ein LLM, um automatisch Fragen zu generieren, die mit dem Dokumentenkorpus beantwortet werden können
- Stellen Sie sicher, dass die Testfälle verschiedene Schwierigkeitsgrade abdecken
- Berücksichtigen Sie bei der Evaluation sowohl quantitative Metriken als auch qualitative Aspekte
- Visualisieren Sie die Ergebnisse, um Stärken und Schwächen des Systems zu identifizieren
