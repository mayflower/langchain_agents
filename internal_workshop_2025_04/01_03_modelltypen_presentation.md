# Modell-Typen im Überblick

## Agenda

- LLM-Modelle im Vergleich
- Embeddings: Grundlagen und Anwendungsfälle
- Multimodale Modelle: Text, Bild, Audio
- Reranker (optional)
- Praxistipps für die Modellauswahl

---

## LLM-Modelle im Vergleich

### OpenAI-Modelle

- **GPT-3.5-Turbo**: Kostengünstig und weit verbreitet
- **GPT-4-Turbo**: Schneller, kostengünstiger als Standard GPT-4
- **GPT-4o**: Multimodal (Text, Bild, Audio)
- **GPT-4o-mini**: Kostengünstigere Version von GPT-4o

### Andere wichtige Modelle

- **Gemini** (Google): Stark multimodal orientiert
- **Claude** (Anthropic): Lange Kontextlänge (bis 100k Token)
- **Mistral/Mixtral**: Open-Source mit guter Performance
- **LLama 3** (Meta): Aktuelle Version mit GPT-4-naher Leistung

---

## Modell-Eigenschaften im Überblick

| Modell  | Stärken                           | Schwächen                       | Kosten |
|---------|-----------------------------------|---------------------------------|--------|
| GPT-3.5 | Schnell, ausgereift               | Begrenzte Komplexität           | €      |
| GPT-4   | Hohe Qualität, gutes Reasoning    | Teurer, langsamer               | €€€    |
| Claude  | Lange Kontexte, ethisch trainiert | Geringere Verfügbarkeit         | €€€    |
| Mistral | Open-Source, selbst hostbar       | Weniger spezialisierte Features | €      |

---

## Embeddings: Grundlagen

- **Definition**: Numerische Vektorrepräsentationen von Text, Bildern oder Daten
- **Dimensionalität**: Typischerweise 768-1536 Dimensionen
- **Eigenschaft**: Ähnliche Konzepte liegen im Vektorraum nahe beieinander

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode("AI ist eine tolle Sache.")
print(tokens)
# [2746, 306, 635, 4258, 3139, 13]

decoded_tokens = [encoding.decode_single_token_bytes(token).decode("utf-8")
                  for token in tokens]
# ['AI', ' ist', ' eine', ' tolle', ' Sache', '.']
```

---

## Embedding-Anwendungsfälle

- **Semantische Suche**: Bedeutung statt nur Schlüsselwörter
- **Clustering**: Gruppierung ähnlicher Konzepte
- **Empfehlungssysteme**: Ähnliche Produkte oder Inhalte finden
- **RAG (Retrieval Augmented Generation)**: Relevante Dokumente für LLMs

Beispiel für Ähnlichkeitsberechnung:

```python
from openai import OpenAI
import numpy as np


# Cosinus-Ähnlichkeit berechnen
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Ähnlichkeit zwischen Embeddings
sim = cosine_similarity(embeddings[0], embeddings[1])
```

---

## Multimodale Modelle

### Unterstützte Modalitäten

- **Text**: Traditionelle Verarbeitung von Sprache
- **Bilder**: Visuelles Verständnis und Generierung
- **Audio**: Spracherkennung und -verarbeitung
- **Video**: Analyse von Videoinhalten (teilweise)

### Wichtige multimodale Modelle

- **GPT-4o** (OpenAI): Text, Bild, Audio
- **Gemini** (Google): Starke multimodale Fähigkeiten
- **Claude 3 Opus** (Anthropic): Text und Bild

---

## Multimodale Anwendungen

- **Bildanalyse und -beschreibung**
  ```python
  vision_llm = llm(model="gpt-4o", max_tokens=1024)
  input = [
      HumanMessage(
          content=[
              "Was ist auf diesem Bild zu sehen?",
              {"image_url": "https://example.com/image.jpg"},
          ]
      )
  ]
  ```

- **Text-zu-Bild-Generierung**
  ```python
  image_url = DallEAPIWrapper(model="dall-e-2").run(
      "Ein Roboter liest in einer Bibliothek"
  )
  ```

---

## Reranker (Optionales Thema)

- **Definition**: Modelle zur präziseren Bewertung der Relevanz zwischen Anfrage
  und Dokumenten
- **Einsatz**: Nach initialer Vektorsuche zur Verfeinerung der Ergebnisse
- **Vorteile**: Verbesserte Genauigkeit, Berücksichtigung des vollständigen
  Textes

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in documents]
scores = reranker.predict(pairs)

# Ergebnisse nach Relevanz sortieren
ranked_results = sorted(zip(documents, scores),
                        key=lambda x: x[1], reverse=True)
```

---

## Entscheidungsmatrix: Modellauswahl

| Anwendungsfall     | Empfohlenes Modell | Begründung                           |
|--------------------|--------------------|--------------------------------------|
| Einfache Chatbots  | GPT-3.5-Turbo      | Kostengünstig, schnell               |
| Reasoning-Aufgaben | GPT-4, Claude      | Bessere Schlussfolgerungsfähigkeiten |
| Bildanalyse        | GPT-4o, Gemini     | Multimodale Fähigkeiten              |
| On-Premise-Lösung  | Llama, Mistral     | Selbst hostbar                       |
| Lange Dokumente    | Claude             | Sehr große Kontextfenster            |

---

## Praxistipps

1. **Komplexität der Aufgabe berücksichtigen**: Einfache Aufgaben benötigen
   keine fortgeschrittenen Modelle
2. **Kosten vs. Leistung abwägen**: Budget und Qualitätsanforderungen
   balancieren
3. **Latenz und Geschwindigkeit testen**: Besonders wichtig für
   Echtzeitanwendungen
4. **Datenschutz beachten**: Wo werden Daten verarbeitet? Welche
   Compliance-Anforderungen gibt es?
5. **Kontextlänge prüfen**: Maximale Textmenge, die verarbeitet werden kann
6. **Multimodale Anforderungen identifizieren**: Werden verschiedene Datentypen
   benötigt?

---

## Praktische Übungen

1. **Token-Analyse**: Wie werden verschiedene Texte in Tokens zerlegt?
2. **Embedding-Vergleich**: Semantische Ähnlichkeit zwischen Konzepten
   visualisieren
3. **Multimodale Anwendung**: Bild generieren und mit demselben LLM analysieren
4. **Modellvergleich**: Dieselbe Aufgabe mit verschiedenen Modellen ausführen
   und vergleichen

---

## Weiterführende Ressourcen

- [OpenAI Models Documentation](https://platform.openai.com/docs/models/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude/docs)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Fragen?
