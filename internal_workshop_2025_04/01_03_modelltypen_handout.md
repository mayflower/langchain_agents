# Modell-Typen im Überblick

## 1. LLM-Modelle im Vergleich

Large Language Models (LLMs) sind das Fundament vieler KI-Anwendungen. Es gibt
verschiedene Anbieter und Modelltypen mit unterschiedlichen Eigenschaften:

### OpenAI-Modelle

- **GPT-3.5-Turbo**: Das kostengünstigste und am weitesten verbreitete Modell
- **GPT-4-Turbo**: Verbesserte Variante (schneller, kostengünstiger als Standard
  GPT-4)
- **GPT-4o**: Neueste multimodale Variante, kann Text, Bild und Audio
  verarbeiten
- **GPT-4o-mini**: Kostengünstigere, kleinere Version des GPT-4o

### Andere bekannte Modelle

- **Gemini** (Google): Starker Fokus auf multimodale Anwendungen
- **Claude** (Anthropic): Bekannt für lange Kontextlänge und ethisch
  ausgerichtetes Training
- **Mistral** und **Mixtral**: Open-Source-Modelle mit guter Performance
- **LLama 3** (Meta): Aktuelles Modell, das in Bereichen an GPT-4 heranreicht

### Kostenaspekte und Einsatzgebiete

| Modell  | Stärken                               | Schwächen                                       | Kostenindikator |
|---------|---------------------------------------|-------------------------------------------------|-----------------|
| GPT-3.5 | Kostengünstig, schnell                | Geringere Qualität bei komplexen Aufgaben       | €               |
| GPT-4   | Hohe Qualität, gutes Reasoning        | Teurer, langsamer                               | €€€             |
| Claude  | Lange Kontextfenster (bis 100k Token) | Geringere Verfügbarkeit                         | €€€             |
| Mistral | Open-Source, selbst hostbar           | Geringere Qualität bei spezialisierten Aufgaben | €               |

## 2. Embeddings

Embeddings sind numerische Vektorrepräsentationen von Text, Bildern oder anderen
Daten, die deren semantische Eigenschaften in einem mehrdimensionalen Raum
abbilden.

### Grundlagen

- Embeddings wandeln Text in hochdimensionale Vektoren um (z.B. 1536 Dimensionen
  bei OpenAI)
- Ähnliche Konzepte liegen im Vektorraum nahe beieinander
- Ermöglichen semantische Suche und Ähnlichkeitsvergleiche

### Anwendungsfälle

- **Semantische Suche**: Finden von Dokumenten basierend auf Bedeutung statt
  Schlüsselwörtern
- **Clustering**: Gruppierung ähnlicher Texte oder Konzepte
- **Recommendation Systems**: Empfehlung ähnlicher Produkte oder Inhalte
- **Retrieval Augmented Generation (RAG)**: Abrufen relevanter Informationen für
  präzisere LLM-Antworten

### Code-Beispiel: Einfache Embedding-Berechnung

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode("AI ist eine tolle Sache.")
print(tokens)
decoded_tokens = [encoding.decode_single_token_bytes(token).decode("utf-8") for
                  token in tokens]
for token in decoded_tokens:
    print(token)
```

### Vektorähnlichkeit berechnen

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Embeddings für verschiedene Sätze berechnen
texts = [
    "Künstliche Intelligenz verändert die Welt",
    "KI revolutioniert alle Wirtschaftsbereiche",
    "Katzen sind niedliche Haustiere"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-ada-002"
)

# Embeddings extrahieren
embeddings = [np.array(item.embedding) for item in response.data]


# Cosinus-Ähnlichkeit berechnen
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Ähnlichkeiten zwischen Sätzen anzeigen
print(
    f"Ähnlichkeit zwischen Satz 1 und 2: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(
    f"Ähnlichkeit zwischen Satz 1 und 3: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
```

## 3. Multimodale Modelle

Multimodale Modelle können mehrere Arten von Eingaben verarbeiten und
miteinander verknüpfen.

### Unterstützte Modalitäten

- **Text**: Traditionelle Textverarbeitung
- **Bilder**: Bildanalyse und -beschreibung
- **Audio**: Spracherkennung und -verarbeitung
- **Video**: Analyse von Videoinhalten (teilweise bereits möglich)

### Wichtige multimodale Modelle

- **GPT-4o** (OpenAI): Integriert Text, Bild und Audio
- **Gemini** (Google): Starker Fokus auf multimodale Eingaben
- **Claude 3 Opus** (Anthropic): Unterstützt Text und Bild

### Anwendungsbeispiele

- **Bildanalyse**: Beschreibung von Bildinhalten, Extraktion von Text aus
  Bildern
- **Visuelle Frage-Antwort**: Beantwortung von Fragen zu Bildinhalten
- **Multimediale Inhaltsmoderation**: Erkennung problematischer Inhalte in
  verschiedenen Medientypen
- **Barrierefreie Anwendungen**: Bildbeschreibungen für sehbehinderte Menschen

### Code-Beispiel: Bildanalyse mit GPT-4o

```python
from openai import OpenAI
from IPython.display import Image
import base64

client = OpenAI()


# Funktion zum Einlesen eines Bildes als Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Bild einlesen
image_path = "example_image.jpg"
base64_image = encode_image(image_path)

# Anfrage an das Modell
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Was ist auf diesem Bild zu sehen?"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)
```

## 4. Reranker (Optionales Thema)

Reranker-Modelle verbessern die Qualität von Suchergebnissen durch eine
präzisere Bewertung der Relevanz.

### Funktionsweise

- Reranker bewerten die Relevanz zwischen einer Anfrage und potenziellen
  Dokumenten
- Sie werden nach einer initialen Suche (z.B. mit Vektorähnlichkeit) eingesetzt
- Sie analysieren den vollständigen Text, nicht nur Vektoren

### Anwendungsfälle

- **Verbesserte RAG-Systeme**: Höhere Genauigkeit bei der Dokumentenauswahl
- **Suchmaschinen**: Präzisere Sortierung von Suchergebnissen
- **Question-Answering**: Bessere Identifikation relevanter Passagen

### Bekannte Reranker-Modelle

- **Cohere Rerank**: Spezialisiertes Modell für Neuordnung von Suchergebnissen
- **BGE Reranker**: Open-Source-Alternative mit guter Performance
- **Cross-Encoder-Modelle** (verschiedene): Auf HuggingFace verfügbare Modelle

### Code-Beispiel: Einfacher Reranking-Prozess

```python
from sentence_transformers import CrossEncoder

# Beispiel-Dokumente
documents = [
    "Berlin ist die Hauptstadt von Deutschland und hat etwa 3,7 Millionen Einwohner.",
    "Paris ist die Hauptstadt von Frankreich und ein beliebtes Reiseziel.",
    "Der Fluss Spree fließt durch Berlin und ist wichtig für die Stadt.",
    "Berlin hat viele Sehenswürdigkeiten wie das Brandenburger Tor."
]

# Abfrage
query = "Was ist die Hauptstadt von Deutschland?"

# Initialer Durchlauf (vereinfacht)
# In der Praxis würde hier eine Vektorsuche stattfinden

# Reranking mit einem Cross-Encoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in documents]
scores = reranker.predict(pairs)

# Ergebnisse sortieren
ranked_results = sorted(zip(documents, scores), key=lambda x: x[1],
                        reverse=True)

# Top-Ergebnisse anzeigen
for doc, score in ranked_results:
    print(f"Score: {score:.4f}, Document: {doc}")
```

## 5. Praktische Überlegungen zur Modellauswahl

Bei der Auswahl des richtigen Modells für Ihre Anwendung sollten Sie folgende
Faktoren berücksichtigen:

1. **Komplexität der Aufgabe**: Einfache Aufgaben benötigen keine
   fortgeschrittenen Modelle
2. **Kosten vs. Leistung**: Abwägung zwischen Budgetbegrenzungen und benötigter
   Qualität
3. **Latenz und Geschwindigkeit**: Reaktionszeit der Modelle in Ihrer Anwendung
4. **Datenschutz und Compliance**: Wo werden die Daten verarbeitet, welche
   Garantien gibt es?
5. **Kontextlänge**: Maximale Textmenge, die das Modell verarbeiten kann
6. **Multimodale Anforderungen**: Werden verschiedene Datentypen verarbeitet?
7. **Integrationsaufwand**: Wie gut lässt sich das Modell in Ihre bestehende
   Infrastruktur integrieren?

### Entscheidungsmatrix: Wann welches Modell?

| Anwendungsfall                | Empfohlenes Modell | Begründung                             |
|-------------------------------|--------------------|----------------------------------------|
| Einfache Chatbots             | GPT-3.5-Turbo      | Kostengünstig, ausreichende Qualität   |
| Komplexe Reasoning-Aufgaben   | GPT-4, Claude      | Bessere Schlussfolgerungsfähigkeiten   |
| Dokumentenanalyse mit Bildern | GPT-4o, Gemini     | Multimodale Fähigkeiten                |
| On-Premise-Lösung             | Llama, Mistral     | Selbst hostbar, keine API-Abhängigkeit |
| Umfangreiche Dokumente        | Claude             | Sehr große Kontextfenster              |

## Quellen und weiterführende Ressourcen

- [OpenAI Models Documentation](https://platform.openai.com/docs/models/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude/docs)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Sentence Transformers Documentation](https://www.sbert.net/)
