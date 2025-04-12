# Vektordatenbanken - Übungen

## Übung 1: Embedding-Vergleich

**Aufgabe:** Erstellen Sie Embeddings für 4 verschiedene Sätze und analysieren
Sie deren semantische Ähnlichkeit.

1. Erstellen Sie eine Python-Funktion, die die Kosinus-Ähnlichkeit zwischen zwei
   Embeddings berechnet
2. Verwenden Sie folgende Sätze:
    - "Vektordatenbanken speichern Embeddings für schnelle Suche."
    - "Datenbanken für Vektoren ermöglichen effiziente Ähnlichkeitsabfragen."
    - "Maschinelles Lernen basiert auf mathematischen Algorithmen."
    - "Berlin ist die Hauptstadt von Deutschland."
3. Berechnen Sie die Ähnlichkeit aller Satzpaare und stellen Sie die Ergebnisse
   in einer Matrix dar
4. Interpretieren Sie die Ergebnisse: Welche Sätze sind semantisch ähnlich,
   welche nicht?

**Tipps:**

- Nutzen Sie die `OpenAIEmbeddings` Klasse und `cosine` aus
  `scipy.spatial.distance`
- Für die Matrix-Darstellung können Sie die Bibliothek `seaborn` verwenden

## Übung 2: Optimale Chunk-Größe ermitteln

**Aufgabe:** Experimentieren Sie mit verschiedenen Chunk-Größen und deren
Auswirkung auf die Retrieval-Qualität.

1. Verwenden Sie den folgenden längeren Text (oder einen eigenen Text Ihrer
   Wahl):

   Vektordatenbanken sind spezialisierte Systeme zur effizienten Speicherung und
Abfrage von Vektordaten. Sie werden hauptsächlich für Ähnlichkeitssuche in
hochdimensionalen Räumen verwendet. Im Gegensatz zu relationalen Datenbanken,
die für strukturierte Abfragen optimiert sind, können Vektordatenbanken schnell
die ähnlichsten Vektoren zu einer gegebenen Abfrage finden.

Die wichtigsten Algorithmen für Vektordatenbanken sind Approximate Nearest
Neighbor (ANN) Algorithmen wie HNSW (Hierarchical Navigable Small World), IVF (
Inverted File Index) und Annoy. Diese Algorithmen erstellen Indizes, die es
ermöglichen, die Suchzeit drastisch zu reduzieren, indem sie einen Kompromiss
zwischen Genauigkeit und Geschwindigkeit eingehen.

Moderne Vektordatenbanken bieten zusätzlich Funktionen wie Metadaten-Filterung,
Clustering und Verwaltung von Vektorsammlungen. Einige populäre
Implementierungen sind Chroma, Qdrant, Pinecone, Weaviate und Milvus. Die Wahl
der richtigen Vektordatenbank hängt von Faktoren wie Datenvolumen, Anforderungen
an Latenz, Skalierbarkeit und Integration mit bestehenden Systemen ab.

2. Teilen Sie den Text mit verschiedenen Chunk-Größen auf:
    - 100 Zeichen
    - 200 Zeichen
    - 500 Zeichen
    - Ganzer Text ohne Aufteilung

3. Erstellen Sie für jede Chunk-Variante eine separate Vektordatenbank und
   führen Sie die gleiche Suchanfrage aus
    - Beispiel: "Wie unterscheiden sich Vektordatenbanken von relationalen
      Datenbanken?"

4. Vergleichen Sie die Ergebnisse: Welche Chunk-Größe liefert die besten
   Ergebnisse?

**Tipps:**

- Nutzen Sie `RecursiveCharacterTextSplitter` mit unterschiedlichen `chunk_size`
  Parametern
- Implementieren Sie einen Hilfsmethode, die den Prozess für alle Chunk-Größen
  wiederholt
- Betrachten Sie sowohl die Qualität der gefundenen Passagen als auch deren
  Kontext

## Übung 3: RAG-Implementierung mit eigenen Dokumenten

**Aufgabe:** Erstellen Sie eine vollständige RAG-Pipeline für ein eigenes
Dokument oder eine kleine Dokumentensammlung.

1. Wählen Sie ein Dokument Ihrer Wahl (z.B. eine Produktdokumentation, ein
   wissenschaftlicher Artikel oder ein Handbuch)
2. Laden Sie das Dokument mit einem geeigneten Document Loader (z.B.
   `PyPDFLoader` für PDFs)
3. Teilen Sie das Dokument in sinnvolle Chunks auf
4. Speichern Sie die Chunks in einer Vektordatenbank
5. Implementieren Sie eine RAG-Kette, die Anfragen zu Ihrem Dokument beantworten
   kann
6. Testen Sie Ihre Implementierung mit 3-5 verschiedenen Fragen

**Tipps:**

- Experiment mit verschiedenen Chunk-Größen und Überlappungen
- Nutzen Sie die Metadaten, um Informationen über Quellen zu speichern
- Probieren Sie verschiedene Retrievers und LLMs aus

## Übung 4: Metadaten-Filterung implementieren

**Aufgabe:** Erweitern Sie eine Vektordatenbank um ein Metadaten-Filtersystem.

1. Erstellen Sie ein Set von mindestens 10 Dokumenten mit strukturierten
   Metadaten, z.B.:
    - Kategorie (tech, business, science, etc.)
    - Quelle (website, report, news, etc.)
    - Datum (in Format YYYY-MM-DD)
    - Bewertung (Zahl zwischen 1-5)

2. Speichern Sie diese Dokumente in einer Vektordatenbank

3. Implementieren Sie Abfragen mit Metadatenfilterung für folgende Szenarien:
    - Nur Dokumente aus einer bestimmten Kategorie
    - Nur Dokumente mit einer Mindestbewertung
    - Nur Dokumente nach einem bestimmten Datum
    - Kombinierte Filter: Kategorie UND Mindestbewertung

4. Vergleichen Sie die Ergebnisse mit und ohne Filterung

**Tipps:**

- Verwenden Sie die `similarity_search`-Methode mit dem `filter` Parameter
- Bauen Sie eine dynamische Filtermethode, die verschiedene Filter unterstützt
- Nutzen Sie Chroma oder Qdrant, die gute Filterunterstützung bieten

## Übung 5: Erweiterte Retrieval-Techniken

**Aufgabe:** Vergleichen Sie die Leistung verschiedener fortgeschrittener
Retrieval-Methoden.

1. Erstellen Sie eine Dokumentensammlung aus mindestens 20 Dokumenten zu einem
   konsistenten Thema
2. Implementieren Sie folgende Retrieval-Techniken:
    - Basisvariante: Einfache Vektorsuche
    - MultiQueryRetriever: Generiert mehrere Varianten der Abfrage
    - ContextualCompressionRetriever: Extrahiert relevante Teile aus Dokumenten
    - Maximum Marginal Relevance (MMR): Fördert Diversität in den Ergebnissen

3. Evaluieren Sie die Qualität der Ergebnisse für 3-5 verschiedene Testfragen
   mit:
    - Anzahl relevanter Dokumente in den Top-5
    - Genauigkeit einer auf den Ergebnissen basierenden LLM-Antwort

4. Dokumentieren Sie, welche Methode für welche Art von Fragen am besten
   funktioniert

**Tipps:**

- Nutzen Sie `LLMChainExtractor` für ContextualCompressionRetriever
- Für MMR können Sie `max_marginal_relevance_search` mit verschiedenen
  `lambda_mult` Werten testen
- Verwenden Sie zur Bewertung das LLM-as-Judge-Konzept mit einem separaten LLM

## Bonus-Übung: Hybride Suche implementieren

**Aufgabe:** Implementieren Sie eine hybride Suchfunktion, die Vektorähnlichkeit
mit Keyword-Suche kombiniert.

1. Erstellen Sie eine Sammlung von Dokumenten
2. Implementieren Sie eine Vektorsuche mit Embeddings
3. Implementieren Sie eine Keyword-Suche (z.B. mit BM25 oder einfachen
   Wortübereinstimmungen)
4. Kombinieren Sie beide Ansätze:
    - Führen Sie beide Suchen getrennt durch
    - Erstellen Sie ein Scoring-System, das die Ergebnisse beider Methoden
      gewichtet kombiniert
    - Sortieren Sie die Ergebnisse nach dem kombinierten Score

5. Vergleichen Sie die Ergebnisse der hybriden Suche mit den Einzelmethoden

**Tipps:**

- Für Keyword-Suche können Sie die Bibliothek `rank_bm25` verwenden
- Ein einfacher Ansatz zur Gewichtung wäre:
  `score = alpha * vector_score + (1-alpha) * keyword_score`
- Experimentieren Sie mit verschiedenen Gewichtungen
