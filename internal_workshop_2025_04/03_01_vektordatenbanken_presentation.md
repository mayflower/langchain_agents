# Vektordatenbanken

## Grundlagen von Vektordatenbanken

- **Definition**: Spezialisierte Datenbanken zum Speichern und Abfragen von Vektoren (numerische Darstellungen von Objekten)
- **Hauptvorteil**: Effiziente Ähnlichkeitssuche in hochdimensionalen Räumen
- **Anwendungsfall für LLMs**: Speicherung von Text-Embeddings für semantische Suche und RAG (Retrieval Augmented Generation)

## Embeddings und Vektorähnlichkeit

- **Embeddings**: Numerische Repräsentationen von Texten, Bildern oder anderen Daten
- **Dimension**: Typischerweise 768 bis 1536 Dimensionen je nach Modell
- **Ähnlichkeitsmaße**:
  - Kosinus-Ähnlichkeit: Winkel zwischen Vektoren (1 = identisch, 0 = unabhängig, -1 = gegensätzlich)
  - Euklidischer Abstand: Direkte Entfernung zwischen Vektoren im Raum
  - Dot-Product: Skalarprodukt der Vektoren

## Vektordatenbanken im Vergleich

| Datenbank | Typ | Besonderheiten | Anwendungsfälle |
|-----------|-----|----------------|-----------------|
| Chroma    | Eingebettet/Server | Einfache Einrichtung, Python-nativ | Prototyping, kleinere Projekte |
| Qdrant    | Server | Hochskalierbar, Filtering | Produktionsanwendungen |
| Pinecone  | Cloud-Service | Vollständig verwaltet, schnell | Enterprise-Anwendungen |
| Weaviate  | Server | GraphQL-Schnittstelle, multimodal | Komplexe Datenstrukturen |
| Milvus    | Server | Hochleistungsfähig, Cloud-nativ | Große Datensätze |

## RAG-Implementierungsstrategien

- **Basic RAG**: Direkte Ähnlichkeitssuche und Einbindung in Prompt
- **Optimierungen**:
  - Query-Transformation: Umformulierung der Abfrage für bessere Ergebnisse
  - Re-Ranking: Nachträgliche Sortierung der Ergebnisse
  - Chunking-Strategien: Optimale Aufteilung der Dokumente
  - Metadaten-Filterung: Einschränkung der Suche basierend auf Metadaten

## Vektorfluss-Optimierung

- **Indexierungsstrategien**: HNSW, IVF, ANNOY für schnellere Suche
- **Query-Augmentation**: Erweiterung der Anfrage mit zusätzlichen Informationen
- **Hybrid Search**: Kombination von Vektorsuche mit Keyword-Suche
- **Multivektorielle Ansätze**: Multiple Embeddings pro Dokument

## LangChain-Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Erstellen oder Laden einer Vektordatenbank
db = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

# Ähnlichkeitssuche
results = db.similarity_search("Beispielanfrage", k=3)

# RAG mit LangChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=db.as_retriever()
)

answer = qa_chain.invoke("Was ist Vektordatenbank?")
