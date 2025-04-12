# Evaluationsmethoden für LLM-Anwendungen - Handout

## 1. LLM-As-A-Judge: Grundlagen und Implementierung

LLM-As-A-Judge nutzt ein LLM zur Bewertung der Ausgaben anderer LLMs nach festgelegten Kriterien.

### Implementierung mit LangChain

```python
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

# Evaluator-LLM initialisieren
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Evaluator mit spezifischen Kriterien erstellen
qa_evaluator = load_evaluator("labeled_criteria", criteria={
    "relevanz": "Ist die Antwort relevant für die Frage?",
    "korrektheit": "Enthält die Antwort korrekte Informationen?",
    "vollständigkeit": "Beantwortet die Antwort die Frage vollständig?",
    "klarheit": "Ist die Antwort klar und verständlich?"
}, llm=evaluator_llm)

# Bewertung einer Antwort
result = qa_evaluator.evaluate_strings(
    prediction="Berlin ist die Hauptstadt von Deutschland und hat etwa 3,7 Millionen Einwohner.",
    input="Was ist die Hauptstadt von Deutschland?"
)

# Ergebnisse ausgeben
for criterion, rating in result["criteria"].items():
    print(f"{criterion}: {rating['rating']} - {rating['reasoning']}")
```

### Vergleichende Bewertung

Für direkten Vergleich zweier Antworten:

```python
pairwise_evaluator = load_evaluator("pairwise_string")

comparison = pairwise_evaluator.evaluate_string_pairs(
    prediction_a="Antwort A...",
    prediction_b="Antwort B...",
    input="Was ist die Hauptstadt von Deutschland?"
)

print(f"Bevorzugte Antwort: {comparison['preferred']}")
print(f"Begründung: {comparison['reasoning']}")
```

## 2. NLP-basierte Testkriterien

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Bewertet die Überlappung zwischen generiertem Text und Referenztext.

```python
from rouge_score import rouge_scorer

# ROUGE-Scorer initialisieren
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Referenz- und Kandidatentext
reference = "Berlin ist die Hauptstadt Deutschlands."
candidate = "Die Hauptstadt von Deutschland ist Berlin."

# ROUGE-Scores berechnen
scores = scorer.score(reference, candidate)

# Ergebnisse ausgeben
for metric, score in scores.items():
    print(f"{metric}: Precision = {score.precision:.4f}, Recall = {score.recall:.4f}, F1 = {score.fmeasure:.4f}")
```

### BLEU (Bilingual Evaluation Understudy)

Misst die Präzision von N-Grammen zwischen generiertem Text und Referenztext.

```python
import nltk
from nltk.translate.bleu_score import sentence_bleu

# NLTK-Daten herunterladen (falls nötig)
nltk.download('punkt')

# Tokenisieren der Texte
reference_tokens = nltk.word_tokenize(reference.lower())
candidate_tokens = nltk.word_tokenize(candidate.lower())

# BLEU-Score berechnen
bleu = sentence_bleu([reference_tokens], candidate_tokens)
print(f"BLEU-Score: {bleu:.4f}")
```

## 3. PI Scrubbing und Datenschutz mit Microsoft Presidio

Presidio ist ein Open-Source-Tool für die Erkennung und Anonymisierung personenbezogener Daten (PII).

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Analyzer und Anonymizer initialisieren
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Beispieltext mit personenbezogenen Daten
text_mit_pii = """
Sehr geehrter Herr Schmidt,
vielen Dank für Ihre Anfrage. Bitte kontaktieren Sie mich unter meiner E-Mail max.mustermann@example.com 
oder telefonisch unter +49 176 12345678.
Ihr Kundenkonto mit der IBAN DE89 3704 0044 0532 0130 00 wurde aktualisiert.
"""

# PII erkennen (deutsche Sprache)
results = analyzer.analyze(text=text_mit_pii, language="de")

# Text anonymisieren
anonymized_text = anonymizer.anonymize(
    text=text_mit_pii,
    analyzer_results=results
)

print("Anonymisierter Text:")
print(anonymized_text.text)
```

## 4. RAG-spezifische Evaluierungsmethoden

RAG-Systeme erfordern die Bewertung mehrerer Komponenten: Retrieval, Kontext und Generierung.

### Kontext-Relevanz bewerten

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Prompt für Kontext-Relevanz erstellen
context_relevance_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Du bist ein Experte für die Bewertung von Retrieval-Systemen."),
    HumanMessage(content="""Bitte bewerte die Relevanz der folgenden Dokumente für die gegebene Anfrage. 
    Gib jedem Dokument eine Bewertung von 1-5 (1 = irrelevant, 5 = hochrelevant) und eine kurze Begründung.
    
    Anfrage: {query}
    
    Zurückgegebene Dokumente:
    {context}
    """)
])

# Chain erstellen
context_eval_chain = LLMChain(llm=evaluator_llm, prompt=context_relevance_prompt)

# Bewertung durchführen
eval_result = context_eval_chain.invoke({
    "query": "Was sind die Vorteile von Vektordatenbanken?",
    "context": "Dokument 1: Vektordatenbanken sind Datenbanken, die Vektoren speichern...\n..."
})
```

### Treue zum Kontext (Faithfulness) prüfen

```python
# Prompt für Treue zum Kontext erstellen
faithfulness_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Du bist ein Experte für die Bewertung von KI-generierten Antworten."),
    HumanMessage(content="""Bitte analysiere, ob die folgende Antwort durch den gegebenen Kontext unterstützt wird. 
    Identifiziere alle Behauptungen in der Antwort und bewerte, ob sie im Kontext enthalten sind oder Halluzinationen darstellen.
    
    Kontext:
    {context}
    
    Antwort:
    {answer}
    """)
])

# Chain erstellen
faithfulness_chain = LLMChain(llm=evaluator_llm, prompt=faithfulness_prompt)

# Bewertung durchführen
faith_eval = faithfulness_chain.invoke({
    "context": "Kontext über Vektordatenbanken...",
    "answer": "Generierte Antwort über Vektordatenbanken..."
})
```

### RAGAS-Metriken implementieren

RAGAS (Retrieval Augmented Generation Assessment) ist ein Framework zur ganzheitlichen Bewertung von RAG-Systemen.

```python
# Vereinfachte RAGAS-Implementierung
ragas_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Du bist ein Experte für die Evaluierung von RAG-Systemen."),
    HumanMessage(content="""Bitte evaluiere das folgende RAG-System nach diesen RAGAS-Metriken:
    
    1. Kontext-Relevanz: Sind die abgerufenen Dokumente relevant für die Anfrage?
    2. Treue (Faithfulness): Wird die Antwort durch den Kontext unterstützt?
    3. Antwort-Relevanz: Beantwortet die generierte Antwort die ursprüngliche Frage?
    4. Kontext-Nutzung: Wie gut nutzt die Antwort den bereitgestellten Kontext?
    
    Anfrage: {query}
    Abgerufene Dokumente: {context}
    Generierte Antwort: {answer}
    
    Gib für jede Metrik eine Bewertung von 0-1 (0 = schlecht, 1 = hervorragend) und berechne einen Gesamtscore.
    """)
])

ragas_chain = LLMChain(llm=evaluator_llm, prompt=ragas_prompt)
```

## 5. Systematische Evaluierung mit Testdatensätzen

```python
# Beispiel für einen einfachen Evaluierungsworkflow
def evaluate_rag_system(test_cases, retrieval_function, generation_function):
    results = []
    
    for test_case in test_cases:
        query = test_case["query"]
        expected = test_case.get("expected_answer")
        
        # 1. Retrieval (Dokumente abrufen)
        retrieved_docs = retrieval_function(query)
        
        # 2. Generation (Antwort generieren)
        answer = generation_function(query, retrieved_docs)
        
        # 3. Evaluierung
        # a) Kontext-Relevanz bewerten
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        context_eval = context_eval_chain.invoke({"query": query, "context": context})
        
        # b) Treue zum Kontext prüfen
        faith_eval = faithfulness_chain.invoke({"context": context, "answer": answer})
        
        # c) ROUGE-Score (falls Referenzantwort vorhanden)
        rouge_score = None
        if expected:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(expected, answer)['rougeL'].fmeasure
        
        # Ergebnisse sammeln
        results.append({
            "query": query,
            "answer": answer,
            "context_evaluation": context_eval["text"],
            "faithfulness_evaluation": faith_eval["text"],
            "rouge_score": rouge_score
        })
    
    return results
```

## 6. Best Practices für Evaluierung

1. **Kombination von Methoden**: Automatische Metriken + LLM-Bewertung + Menschliches Feedback
2. **Kontinuierliche Evaluierung**: Teil des Entwicklungsprozesses, nicht nur am Ende
3. **Repräsentative Testdaten**: Decken reale Anwendungsfälle ab
4. **Datenschutz**: PII-Scrubbing für alle Evaluierungsdaten
5. **Domänenspezifische Kriterien**: Angepasst an den Anwendungsfall
6. **Ergebnisdokumentation**: Verbesserungen über die Zeit verfolgen
7. **Menschliche Überprüfung**: Besonders bei kritischen Anwendungen
8. **A/B-Tests**: Für Modelloptimierung und Prompt-Engineering
