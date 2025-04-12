# Vektordatenbanken - Handout

## 1. Grundlagen von Vektordatenbanken

Vektordatenbanken sind spezialisierte Datenbanken, die für die effiziente Speicherung und Abfrage von Vektoren (numerischen Darstellungen von Objekten) optimiert sind. Im Kontext von LLMs werden sie hauptsächlich für die Speicherung von Text-Embeddings verwendet.

### Wichtige Konzepte:

- **Embedding**: Eine numerische Repräsentation (Vektor) von Text oder anderen Daten in einem hochdimensionalen Raum
- **Ähnlichkeitssuche**: Finden von Vektoren, die einem Abfragevektor am ähnlichsten sind
- **Dimensionalität**: Typischerweise 768-1536 Dimensionen pro Vektor bei modernen Embedding-Modellen

## 2. Embeddings und Vektorähnlichkeit

### Embedding-Erstellung:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = "Dies ist ein Beispieltext."
vector = embeddings.embed_query(text)
# vector ist nun ein Array mit 1536 Floats (für text-embedding-ada-002)
```

### Ähnlichkeitsmaße:

- **Kosinus-Ähnlichkeit**: Misst den Kosinus des Winkels zwischen zwei Vektoren
  ```python
  import numpy as np
  from scipy.spatial.distance import cosine
  
  similarity = 1 - cosine(vector1, vector2)  # 1 = identisch, 0 = unabhängig
  ```

- **Euklidischer Abstand**: Direkte Distanz zwischen Vektoren
  ```python
  distance = np.linalg.norm(vector1 - vector2)  # Kleinere Werte = ähnlicher
  ```

## 3. Populäre Vektordatenbanken

### Chroma

Einfache, eingebettete Vektordatenbank, ideal für Prototyping:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Dokumente laden und aufteilen
loader = TextLoader("dokument.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Vektordatenbank erstellen
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Optional: Pfad zum Speichern
)

# Suche durchführen
results = db.similarity_search("Meine Frage?", k=3)
```

### Qdrant

Skalierbare, produktionsreife Vektordatenbank:

```python
from langchain_community.vectorstores import Qdrant
import qdrant_client

# Client erstellen
client = qdrant_client.QdrantClient(url="http://localhost:6333")

# Vektordatenbank erstellen
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url="http://localhost:6333",
    collection_name="meine_sammlung",
)

# Suche mit Metadaten-Filter
results = qdrant.similarity_search_with_score(
    "Meine Frage?",
    k=3,
    filter={"metadata_field": "filter_value"}
)
```

## 4. RAG-Implementierungsstrategien

### Einfaches RAG-Beispiel:

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# RAG-Kette erstellen
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

# Anfrage stellen
result = qa_chain.invoke("Was sind die Hauptvorteile von Vektordatenbanken?")
print(result["result"])
```

### Erweiterte Strategien:

- **Chunk-Größe optimieren**: Kleinere Chunks (150-500 Tokens) für präzisere Antworten, größere Chunks (800-1500 Tokens) für mehr Kontext
- **Überlappung einsetzen**: 10-20% Überlappung um Kontext über Chunk-Grenzen hinweg zu erhalten
- **Metadaten anreichern**: Titel, Abschnittsnummern, Quellangaben als Metadaten speichern
- **Hybrid-Suche**: Vektorsuche mit Keyword-Suche kombinieren

## 5. Vektorfluss-Optimierung

### Query-Transformation:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Erzeugt mehrere Varianten der Suchanfrage
retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=ChatOpenAI()
)

# Verbesserte Suche mit mehreren Anfragevarianten
results = retriever.get_relevant_documents("Welche Vektordatenbank ist am besten?")
```

### Re-Ranking:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Re-Ranker erstellen, der nur relevante Teile aus den Dokumenten extrahiert
compressor = LLMChainExtractor.from_llm(ChatOpenAI())
compression_retriever = ContextualCompressionRetriever(
    base_retriever=db.as_retriever(),
    doc_compressor=compressor
)

# Verbesserte Ergebnisse durch Kompression und Re-Ranking
compressed_docs = compression_retriever.get_relevant_documents("Meine spezifische Frage?")
```

## 6. Best Practices

- Optimieren Sie Ihre Chunk-Strategie für Ihre spezifischen Dokumente
- Experimentieren Sie mit verschiedenen Embedding-Modellen
- Nutzen Sie Metadaten zum effizienten Filtern
- Implementieren Sie Caching für häufige Anfragen
- Evaluieren Sie Ihre RAG-Pipeline mit Ground-Truth-Datensätzen
