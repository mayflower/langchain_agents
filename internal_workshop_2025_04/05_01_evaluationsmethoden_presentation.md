# Evaluationsmethoden für LLM-Anwendungen

## Überblick

- **LLM-As-A-Judge**: Bewertung von LLM-Ausgaben durch andere LLMs
- **NLP-basierte Metriken**: ROUGE, BLEU, etc. für quantitative Bewertung
- **PI Scrubbing**: Datenschutz mit Microsoft Presidio
- **RAG-spezifische Evaluation**: Bewertung von Retrieval und Generation

## LLM-As-A-Judge: Grundlagen

- **Konzept**: Ein LLM bewertet die Ausgabe eines anderen LLMs
- **Vorteile**:
  - Skalierbar und kostengünstiger als menschliche Bewerter
  - Konsistente Bewertungen nach festgelegten Kriterien
  - Flexibel für unterschiedliche Bewertungsdimensionen
- **Herausforderungen**:
  - Reproduzierbarkeit je nach Modell variabel
  - Eigene Voreingenommenheit des bewertenden LLMs
  - Kalibrierung mit menschlichen Urteilen notwendig

## LLM-As-A-Judge: Implementierung

- **Festgelegte Kriterien**: Relevanz, Korrektheit, Klarheit, etc.
- **Bewertungsskalen**: Numerische Werte oder kategorische Bewertungen
- **Umsetzung in LangChain**:
  ```python
  evaluator = load_evaluator("labeled_criteria", criteria={
      "relevanz": "Ist die Antwort relevant für die Frage?",
      "korrektheit": "Enthält die Antwort korrekte Informationen?",
      "klarheit": "Ist die Antwort klar und verständlich?"
  }, llm=evaluator_llm)
  
  result = evaluator.evaluate_strings(
      prediction=response,
      input=question
  )
  ```

## Vergleichende Bewertung

- **Paarweise Vergleiche** statt absolute Bewertungen
- **Implementierung**:
  ```python
  pairwise_evaluator = load_evaluator("pairwise_string")
  comparison = pairwise_evaluator.evaluate_string_pairs(
      prediction_a=response_a,
      prediction_b=response_b,
      input=question
  )
  ```
- **Vorteil**: Einfacher für LLMs zu beurteilen als absolute Qualität
- **Eignet sich für**: A/B-Tests, Modellvergleiche, Prompt-Optimierung

## NLP-basierte Metriken: ROUGE

- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
- **Verschiedene Varianten**:
  - **ROUGE-N**: N-Gram-Überlappung (ROUGE-1, ROUGE-2)
  - **ROUGE-L**: Längste gemeinsame Teilsequenz
- **Metriken**: Precision, Recall, F1-Score
- **Implementierung**:
  ```python
  from rouge_score import rouge_scorer
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
  scores = scorer.score(reference, candidate)
  ```

## NLP-basierte Metriken: BLEU

- **BLEU** (Bilingual Evaluation Understudy)
- **Ursprünglich** für maschinelle Übersetzung entwickelt
- **Funktionsweise**: Bewertet Präzision von N-Grammen mit Strafterm für zu kurze Antworten
- **Implementierung**:
  ```python
  from nltk.translate.bleu_score import sentence_bleu
  reference_tokens = nltk.word_tokenize(reference)
  candidate_tokens = nltk.word_tokenize(candidate)
  bleu = sentence_bleu([reference_tokens], candidate_tokens)
  ```
- **Werte**: 0-1, höher ist besser

## Stärken und Schwächen automatischer Metriken

### Stärken
- **Reproduzierbarkeit**: Konsistente Ergebnisse
- **Geschwindigkeit**: Effiziente Bewertung von vielen Antworten
- **Quantifizierbarkeit**: Leicht vergleichbare Zahlen

### Schwächen
- **Semantisches Verständnis**: Erfassen nicht immer die Bedeutung
- **Kreativität**: Bewerten kreative Lösungen oft schlecht
- **Kontextbezug**: Berücksichtigen nicht immer den Gesamtkontext

## PI Scrubbing und Datenschutz

- **PII** (Personally Identifiable Information) muss in Evaluierungsdaten anonymisiert werden
- **Microsoft Presidio**: Open-Source-Lösung für PII-Erkennung und -Anonymisierung
- **Unterstützte PII-Typen**:
  - Namen, E-Mail-Adressen, Telefonnummern
  - Finanzdaten (IBAN, Kreditkartennummern)
  - Persönliche IDs (Sozialversicherungsnummer, Reisepassnummer)
  - Adressen, Geburtsdaten

## PII-Erkennung mit Presidio

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# PII erkennen
results = analyzer.analyze(text=text_mit_pii, language="de")

# Text anonymisieren
anonymized_text = anonymizer.anonymize(
    text=text_mit_pii,
    analyzer_results=results
)
```

## RAG-spezifische Evaluation

- **Herausforderung**: RAG-Systeme haben mehrere zu bewertende Komponenten
- **Bewertungsdimensionen**:
  1. **Kontext-Relevanz**: Relevanz der abgerufenen Dokumente
  2. **Treue (Faithfulness)**: Werden Behauptungen durch Kontext gestützt?
  3. **Antwort-Relevanz**: Beantwortet die Ausgabe die Anfrage?
  4. **Kontext-Nutzung**: Wie gut wird der bereitgestellte Kontext genutzt?

## RAG-Evaluierungstechniken

### Kontext-Relevanz
- Bewertung, ob abgerufene Dokumente für die Anfrage relevant sind
- Frage: "Wurden die richtigen Informationen abgerufen?"

### Treue zum Kontext (Faithfulness)
- Überprüft, ob die Antwort durch den Kontext gestützt wird
- Identifiziert Halluzinationen und nicht unterstützte Behauptungen

### Antwort-Relevanz
- Bewertet, ob die Antwort tatsächlich die Frage beantwortet
- Berücksichtigt Vollständigkeit und Genauigkeit

## RAGAS: Framework für RAG-Evaluation

- **RAGAS**: Retrieval Augmented Generation Assessment
- **Metriken**:
  - **Context Relevancy**: Relevanz der abgerufenen Passagen
  - **Faithfulness**: Übereinstimmung der Antwort mit dem Kontext
  - **Answer Relevancy**: Relevanz der Antwort zur Frage
  - **Context Recall**: Abdeckung der notwendigen Informationen
- **Implementierung**: Sowohl Open-Source-Bibliothek als auch selbst implementierbar

## Systematische Evaluierung mit Testdatensätzen

- **Testdatensatz erstellen**:
  - Repräsentative Fragen aus dem Anwendungsbereich
  - Erwartete Antworten/Referenzantworten
  - Edge Cases und schwierige Szenarien
- **Metriken aggregieren** über alle Testfälle
- **Schwachstellen identifizieren** durch Analyse der schlechtesten Ergebnisse
- **A/B-Tests** für inkrementelle Verbesserungen

## Evaluierungsworkflow

1. **Vorbereitung**:
   - Testdatensatz definieren
   - Evaluierungsmetriken auswählen
   - PII-Scrubbing implementieren

2. **Durchführung**:
   - Antworten für alle Testfälle generieren
   - Automatische Metriken berechnen
   - LLM-basierte Bewertungen durchführen

3. **Analyse**:
   - Ergebnisse aggregieren und visualisieren
   - Schwachstellen identifizieren
   - Verbesserungsmaßnahmen ableiten

## Best Practices für LLM-Evaluation

- **Multiple Methoden kombinieren**: Automatische Metriken + LLM-Bewertung + Menschliches Feedback
- **Kontinuierlich evaluieren**: Teil des Entwicklungsprozesses, nicht nur am Ende
- **Repräsentative Testdaten** verwenden, die reale Anwendungsfälle abdecken
- **Datenschutz beachten**: PII-Scrubbing für alle Evaluierungsdaten
- **Domänenspezifische Kriterien** definieren für Ihren Anwendungsfall
- **Ergebnisse dokumentieren** und Verbesserungen über die Zeit verfolgen

## Zusammenfassung

- **LLM-As-A-Judge**: Skalierbare Evaluation mit definierten Kriterien
- **NLP-Metriken**: Quantitative Bewertung der Textähnlichkeit
- **Datenschutz**: PII-Scrubbing mit Tools wie Presidio
- **RAG-Evaluation**: Bewertung von Retrieval und Generation
- **Systematischer Ansatz**: Testdatensätze, Multi-Metrik-Evaluation, kontinuierliche Verbesserung

## Nächste Schritte

- Evaluierungsframework in Ihre Entwicklungspipeline integrieren
- Eigene Testdatensätze für Ihren spezifischen Anwendungsfall erstellen
- Evaluierungskriterien an Ihre Geschäftsanforderungen anpassen
- A/B-Tests für Modelloptimierung und Prompt-Engineering durchführen
