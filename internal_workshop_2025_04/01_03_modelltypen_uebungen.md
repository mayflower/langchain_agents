# Modell-Typen: Übungen und Lösungen

## Übung 1: Token-Analyse verschiedener Texttypen

**Aufgabe**: Analysieren Sie, wie verschiedene Arten von Text in Tokens zerlegt
werden und vergleichen Sie die Ergebnisse.

**Lösungsansatz**:

```python
import tiktoken


def analyze_tokens(text, model="gpt-4o"):
    """Tokenisiert einen Text und gibt detaillierte Informationen aus"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    decoded_tokens = [encoding.decode_single_token_bytes(token).decode("utf-8")
                      for token in tokens]

    print(f"Text: '{text}'")
    print(f"Anzahl Tokens: {len(tokens)}")
    print("Tokens:")
    for i, token in enumerate(decoded_tokens):
        print(f"  {i + 1}: '{token}'")
    print("\n")

    return len(tokens)


# Verschiedene Texttypen testen
text_types = {
    "Normaler Text": "Künstliche Intelligenz wird unsere Zukunft verändern.",
    "Fachbegriffe": "Transformer-Architektur und Attention-Mechanismen revolutionieren NLP.",
    "Fremdsprache": "Machine learning is transforming various industries worldwide.",
    "Zahlen": "Die Preise liegen zwischen 123,45€ und 678,90€.",
    "Sonderzeichen": "E-Mail: example@domain.com, Tel: +49-123-4567890",
    "JSON-Struktur": '{"name": "Max", "age": 30, "city": "Berlin"}'
}

total_tokens = {}

for label, text in text_types.items():
    print(f"=== {label} ===")
    token_count = analyze_tokens(text)
    total_tokens[label] = token_count

# Vergleichsstatistik
print("=== Zusammenfassung ===")
for label, count in total_tokens.items():
    print(f"{label}: {count} Tokens")
```

## Übung 2: Semantische Ähnlichkeit mit Embeddings

**Aufgabe**: Berechnen Sie Embeddings für verschiedene Sätze aus ähnlichen und
unterschiedlichen Themenbereichen und visualisieren Sie die semantischen
Beziehungen.

**Lösungsansatz**:

```python
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Verschiedene Themenbereiche mit ähnlichen und unterschiedlichen Sätzen
texts = [
    # KI-Bereich
    "Künstliche Intelligenz revolutioniert viele Branchen.",
    "Machine Learning wird in zahlreichen Industrien eingesetzt.",
    "Deep Learning ist ein Teilbereich des maschinellen Lernens.",

    # Sport-Bereich
    "Fußball ist ein beliebter Mannschaftssport weltweit.",
    "Basketball wird in einer Halle mit zwei Körben gespielt.",
    "Sportliche Aktivität fördert die Gesundheit und das Wohlbefinden.",

    # Musik-Bereich
    "Klassische Musik stammt hauptsächlich aus der europäischen Tradition.",
    "Jazz entwickelte sich Anfang des 20. Jahrhunderts in den USA.",
    "Musikinstrumente können in verschiedene Kategorien eingeteilt werden."
]

# Themenbereich für jeden Text zum Plotten
topic_labels = ["KI", "KI", "KI", "Sport", "Sport", "Sport", "Musik", "Musik",
                "Musik"]
topic_colors = ["blue", "blue", "blue", "red", "red", "red", "green", "green",
                "green"]

# Embeddings berechnen
response = client.embeddings.create(
    input=texts,
    model="text-embedding-ada-002"
)

embeddings = [np.array(item.embedding) for item in response.data]


# Cosinus-Ähnlichkeit berechnen
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Ähnlichkeitsmatrix erstellen
similarity_matrix = np.zeros((len(texts), len(texts)))
for i in range(len(texts)):
    for j in range(len(texts)):
        similarity_matrix[i, j] = cosine_similarity(embeddings[i],
                                                    embeddings[j])

# Heatmap der Ähnlichkeiten
plt.figure(figsize=(12, 10))
plt.imshow(similarity_matrix, cmap='viridis')
plt.colorbar(label='Cosinus-Ähnlichkeit')

# Achsenbeschriftungen mit abgekürzten Texten
shortened_texts = [t[:20] + "..." for t in texts]
plt.xticks(range(len(texts)), shortened_texts, rotation=45, ha="right")
plt.yticks(range(len(texts)), shortened_texts)

plt.title(
    'Semantische Ähnlichkeit zwischen Texten verschiedener Themenbereiche')
plt.tight_layout()
plt.savefig('similarity_heatmap.png')

# PCA für 2D-Visualisierung
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 2D-Plot mit Themenbereichen farblich markiert
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(reduced_embeddings):
    plt.scatter(x, y, c=topic_colors[i], label=topic_labels[i], s=100)

# Legende ohne Duplikate
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize=12)

# Textlabels hinzufügen
for i, (x, y) in enumerate(reduced_embeddings):
    plt.annotate(f"{i + 1}: {texts[i][:15]}...", (x, y), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')

plt.title('2D-Projektion der Texte nach semantischer Ähnlichkeit')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('embeddings_2d.png')

# Textbeschreibungen ausgeben
for i, text in enumerate(texts):
    print(f"Text {i + 1} ({topic_labels[i]}): {text}")

# Ähnlichkeitsanalyse ausgeben
print("\nÄhnlichkeitsanalyse:")
# Innerhalb der Themengruppen
for topic in ["KI", "Sport", "Musik"]:
    indices = [i for i, t in enumerate(topic_labels) if t == topic]
    avg_similarity = np.mean(
        [similarity_matrix[i][j] for i in indices for j in indices if i != j])
    print(
        f"Durchschnittliche Ähnlichkeit innerhalb des Themas '{topic}': {avg_similarity:.4f}")

# Zwischen verschiedenen Themengruppen
for topic1 in ["KI", "Sport", "Musik"]:
    for topic2 in ["KI", "Sport", "Musik"]:
        if topic1 >= topic2:
            continue
        indices1 = [i for i, t in enumerate(topic_labels) if t == topic1]
        indices2 = [i for i, t in enumerate(topic_labels) if t == topic2]
        avg_similarity = np.mean(
            [similarity_matrix[i][j] for i in indices1 for j in indices2])
        print(
            f"Durchschnittliche Ähnlichkeit zwischen '{topic1}' und '{topic2}': {avg_similarity:.4f}")
```

## Übung 3: Vergleich verschiedener LLM-Modelle bei identischer Aufgabe

**Aufgabe**: Vergleichen Sie die Antworten und Performance verschiedener
LLM-Modelle bei identischen Aufgaben.

**Lösungsansatz**:

```python
from helpers import llm
import time
import pandas as pd
import matplotlib.pyplot as plt

# Zu testende Modelle
models = {
    "gpt-3.5-turbo": llm(model="gpt-3.5-turbo"),
    "gpt-4o": llm(model="gpt-4o"),
    "gpt-4o-mini": llm(model="gpt-4o-mini"),
    # Weitere Modelle je nach Verfügbarkeit
}

# Verschiedene Testaufgaben
test_prompts = [
    "Erkläre das Konzept der künstlichen Intelligenz in 3 Sätzen.",
    "Schreibe ein kurzes Gedicht über das Programmieren.",
    "Was sind die wichtigsten Unterschiede zwischen Python und JavaScript?",
    "Wie funktioniert ein Transformer-Modell in einfachen Worten?",
    "Gib 5 Tipps zur Optimierung von LLM-Prompts."
]

# Ergebnisse speichern
results = []

for prompt_idx, prompt in enumerate(test_prompts):
    print(f"\n=== Testaufgabe {prompt_idx + 1}: ===")
    print(prompt)

    for model_name, model_instance in models.items():
        print(f"\n-- Modell: {model_name} --")

        # Antwort messen
        start_time = time.time()
        response = model_instance.invoke(prompt)
        end_time = time.time()
        duration = end_time - start_time

        # Antwort ausgeben
        print(f"Antwort: {response.content}")
        print(f"Dauer: {duration:.2f} Sekunden")

        # Ergebnisse speichern
        token_count = len(response.content.split())  # Grobe Schätzung
        results.append({
            "Modell": model_name,
            "Aufgabe": f"Aufgabe {prompt_idx + 1}",
            "Dauer (s)": duration,
            "Antwortlänge (Tokens)": token_count
        })

# Ergebnisse als DataFrame
df = pd.DataFrame(results)

# Visualisierung der Antwortzeiten
plt.figure(figsize=(12, 6))
pivot_duration = df.pivot(index="Aufgabe", columns="Modell", values="Dauer (s)")
pivot_duration.plot(kind="bar")
plt.title("Antwortzeiten verschiedener Modelle")
plt.ylabel("Zeit in Sekunden")
plt.xlabel("Testaufgabe")
plt.legend(title="Modell")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("model_duration_comparison.png")

# Visualisierung der Antwortlängen
plt.figure(figsize=(12, 6))
pivot_length = df.pivot(index="Aufgabe", columns="Modell",
                        values="Antwortlänge (Tokens)")
pivot_length.plot(kind="bar")
plt.title("Antwortlängen verschiedener Modelle")
plt.ylabel("Geschätzte Token-Anzahl")
plt.xlabel("Testaufgabe")
plt.legend(title="Modell")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("model_length_comparison.png")

# Zusammenfassung
print("\n=== Zusammenfassung ===")
summary = df.groupby("Modell").agg({
    "Dauer (s)": ["mean", "min", "max"],
    "Antwortlänge (Tokens)": ["mean", "min", "max"]
})
print(summary)
```

## Übung 4: Multimodale Anwendung: Bild-Text-Interaktion

**Aufgabe**: Entwickeln Sie eine einfache multimodale Anwendung, die Bilder
generiert und sie anschließend mit demselben LLM analysiert.

**Lösungsansatz**:

```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.schema import HumanMessage, StrOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from IPython.display import Image, display
from helpers import llm
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

# 1. Bildgenerierung mit unterschiedlichen Stilen
image_concepts = [
    "Ein Roboter, der in einer Bibliothek liest",
    "Eine futuristische Stadt unter Wasser",
    "Ein Raumschiff, das auf einem fremden Planeten landet"
]

image_styles = [
    "realistisch, fotografisch",
    "comic-stil, bunt",
    "ölgemälde, künstlerisch"
]

# Prompt für optimierte DALL-E Anweisungen
prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "Erstelle einen präzisen Prompt für DALL-E, um ein Bild zu generieren, das folgendes zeigt: "
        "{concept}. Der Stil sollte {style} sein. Sei detailliert und beschreibe Farben, "
        "Perspektive und Stimmung, aber halte es unter 100 Wörtern."
    )
])

chain = prompt_template | llm(temperature=0.7) | StrOutputParser()

# Vision-fähiges LLM für Bildanalyse
vision_llm = llm(model="gpt-4o", max_tokens=1024)

# Bilder erzeugen und analysieren
results = []

for concept in image_concepts:
    for style in image_styles:
        print(f"\n=== Konzept: {concept} | Stil: {style} ===")

        # 1. DALL-E Prompt optimieren
        optimized_prompt = chain.invoke({
            "concept": concept,
            "style": style
        })
        print(f"Optimierter DALL-E Prompt:\n{optimized_prompt}\n")

        try:
            # 2. Bild generieren
            dalle_api = DallEAPIWrapper(model="dall-e-2", size="1024x1024")
            image_url = dalle_api.run(optimized_prompt)
            print(f"Bild generiert: {image_url}")

            # 3. Bild anzeigen
            response = requests.get(image_url)
            img = Image(data=response.content)
            display(img)

            # 4. Bild analysieren
            vision_input = [
                HumanMessage(
                    content=[
                        "Analysiere dieses Bild detailliert: Beschreibe die Hauptelemente, den Stil, "
                        "die Stimmung und ob es das Konzept gut umsetzt. Das Konzept war: "
                        f"'{concept}' im Stil '{style}'.",
                        {"image_url": image_url}
                    ]
                )
            ]

            analysis = vision_llm.invoke(vision_input)
            print("\nBildanalyse:")
            print(analysis.content)

            # Ergebnisse speichern
            results.append({
                "Konzept": concept,
                "Stil": style,
                "DALL-E Prompt": optimized_prompt,
                "Bild-URL": image_url,
                "Analyse": analysis.content
            })

        except Exception as e:
            print(f"Fehler bei der Bildgenerierung oder -analyse: {e}")
            print(
                "Um diesen Teil auszuführen, stellen Sie sicher, dass Sie gültige API-Schlüssel haben.")

# Zusammenfassung
if results:
    print("\n=== Zusammenfassung ===")
    print(f"Insgesamt wurden {len(results)} Bilder generiert und analysiert.")

    # Optional: Ergebnisse in CSV speichern
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv("multimodal_results.csv", index=False)
    print("Ergebnisse wurden in 'multimodal_results.csv' gespeichert.")
```

## Übung 5: Reranker-Integration in eine einfache Suchfunktion

**Aufgabe**: Implementieren Sie eine einfache Dokumentensuche, die sowohl
Embedding-basierte Suche als auch Reranking nutzt, und vergleichen Sie die
Ergebnisse.

**Lösungsansatz**:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

# Beispiel-Dokumentensammlung
documents = [
    "Künstliche Intelligenz (KI) ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens befasst.",
    "Machine Learning ermöglicht es Computern, aus Erfahrungen zu lernen und sich anzupassen, ohne explizit programmiert zu werden.",
    "Deep Learning ist ein Teilbereich des maschinellen Lernens, der auf künstlichen neuronalen Netzen basiert.",
    "Python ist eine beliebte Programmiersprache für KI und Data Science aufgrund ihrer einfachen Syntax und vielen Bibliotheken.",
    "TensorFlow und PyTorch sind zwei der beliebtesten Frameworks für die Entwicklung von Deep-Learning-Modellen.",
    "Natural Language Processing (NLP) konzentriert sich auf die Interaktion zwischen Computern und menschlicher Sprache.",
    "Computer Vision befasst sich mit der Extraktion von Informationen aus Bildern und Videos.",
    "Reinforcement Learning ist eine Methode des maschinellen Lernens, bei der ein Agent durch Interaktion mit seiner Umgebung lernt.",
    "Der Turing-Test wurde von Alan Turing entwickelt, um festzustellen, ob eine Maschine menschenähnliche Intelligenz zeigt.",
    "Big Data bezieht sich auf extrem große Datensätze, die mit herkömmlichen Datenverarbeitungsmethoden nicht mehr bewältigt werden können."
]

# Beispielsuchen
queries = [
    "Was ist künstliche Intelligenz?",
    "Python Programmierung für KI",
    "Deep Learning und neuronale Netze",
    "Wie lernen Computer aus Daten?"
]

# 1. Embedding-Modell laden
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Reranker-Modell laden
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 3. Embeddings für Dokumente vorberechnen
document_embeddings = embedding_model.encode(documents)

results = []

for query in queries:
    print(f"\n=== Suchanfrage: {query} ===")

    # 4. Embedding für die Suchanfrage berechnen
    query_embedding = embedding_model.encode(query)

    # 5. Ähnlichkeiten berechnen (Cosinus-Ähnlichkeit)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # 6. Ergebnisse nach Embedding-Ähnlichkeit sortieren
    embedding_ranking = sorted(zip(range(len(documents)), similarities),
                               key=lambda x: x[1], reverse=True)

    print("\nRanking basierend auf Embedding-Ähnlichkeit:")
    for i, (doc_idx, score) in enumerate(embedding_ranking[:5]):
        print(f"{i + 1}. Score: {score:.4f} - {documents[doc_idx][:100]}...")

    # 7. Reranking mit CrossEncoder
    document_pairs = [[query, documents[doc_idx]] for doc_idx, _ in
                      embedding_ranking]
    reranker_scores = reranker.predict(document_pairs)

    # 8. Ergebnisse nach Reranker-Score sortieren
    reranker_ranking = sorted(
        zip([idx for idx, _ in embedding_ranking], reranker_scores),
        key=lambda x: x[1], reverse=True)

    print("\nRanking nach Reranking:")
    for i, (doc_idx, score) in enumerate(reranker_ranking[:5]):
        print(f"{i + 1}. Score: {score:.4f} - {documents[doc_idx][:100]}...")

    # 9. Ergebnisse für spätere Visualisierung speichern
    for doc_idx, _ in embedding_ranking[:5]:
        # Finde die Position im Reranking
        rerank_position = next(
            (i + 1 for i, (idx, _) in enumerate(reranker_ranking) if
             idx == doc_idx), None)

        # Embedding-Score
        emb_score = next(
            (score for idx, score in embedding_ranking if idx == doc_idx), 0)

        # Reranker-Score
        rerank_score = next(
            (score for idx, score in reranker_ranking if idx == doc_idx), 0)

        results.append({
            "Query": query,
            "Document": documents[doc_idx][:50] + "...",
            "Doc_ID": doc_idx,
            "Embedding_Rank": list(zip(*embedding_ranking))[0].index(
                doc_idx) + 1,
            "Reranker_Rank": rerank_position,
            "Embedding_Score": emb_score,
            "Reranker_Score": rerank_score
        })

# Ergebnisse in DataFrame konvertieren
df = pd.DataFrame(results)

# Visualisierung der Rangänderungen
plt.figure(figsize=(14, 8))

# Gruppieren nach Suchanfrage
for i, (query, group) in enumerate(df.groupby("Query")):
    plt.subplot(2, 2, i + 1)

    # Nach Embedding-Rang sortieren
    sorted_docs = group.sort_values("Embedding_Rank")

    # Verbindungslinien zwischen Embedding-Rang und Reranker-Rang
    for _, row in sorted_docs.iterrows():
        plt.plot([1, 2], [row["Embedding_Rank"], row["Reranker_Rank"]],
                 'o-', markersize=8,
                 label=f"Doc {row['Doc_ID']}" if i == 0 else "")

    plt.title(f"Anfrage: {query}")
    plt.xlim(0.8, 2.2)
    plt.ylim(0.5, 5.5)
    plt.xticks([1, 2], ["Embedding", "Reranker"])
    plt.yticks(range(1, 6))
    plt.gca().invert_yaxis()  # Rang 1 oben anzeigen

    # Nur im ersten Plot eine Legende
    if i == 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("reranking_comparison.png")

# Zusammenfassung: Wie oft hat sich das Ranking geändert?
rank_changes = (df["Embedding_Rank"] != df["Reranker_Rank"]).sum()
print(f"\n=== Zusammenfassung ===")
print(
    f"In {rank_changes} von {len(df)} Fällen ({rank_changes / len(df) * 100:.1f}%) hat das Reranking die Position geändert.")

# Durchschnittliche Rangänderung
avg_rank_change = abs(df["Embedding_Rank"] - df["Reranker_Rank"]).mean()
print(f"Durchschnittliche Rangänderung: {avg_rank_change:.2f} Positionen")

# Top-1 Änderungen (wenn sich das wichtigste Ergebnis geändert hat)
top1_changes = ((df["Embedding_Rank"] == 1) & (
            df["Reranker_Rank"] != 1)).sum() +
               ((df["Embedding_Rank"] != 1) & (df["Reranker_Rank"] == 1)).sum()
print(
    f"In {top1_changes} von {len(df) // 5} Suchen hat sich das Top-Ergebnis geändert.")
```
