# Workshop-Plan: KI-Aufschlau für Entwickler

## Überblick

Dieser Workshop gibt eine praxisorientierte Einführung in die Entwicklung mit
Large Language Models (LLMs) und fokussiert sich auf die Implementierung
moderner KI-Anwendungen mit Frameworks wie LangChain.

**Dauer**: ca. 4 Stunden
**Zielgruppe**: Entwickler mit Python-Grundkenntnissen
**Vorkenntnisse**: Grundlegende Programmierkenntnisse, Interesse an
KI-Anwendungen

## Vorbereitung

Vor dem Workshop sollten die Teilnehmer:

### Option 1: AWS-Instanz nutzen

1. URL https://workshop.aisaisbaby.com/ aufrufen
2. Account anlegen, falls noch nicht vorhanden
3. Mit den Zugangsdaten einloggen

### Option 2: Lokale Installation

1. Das Repository klonen:
   `git clone https://github.com/mayflower/langchain_agents.git`
2. Die Umgebung einrichten:
    - Option 1: VSCode mit DevContainers nutzen
    - Option 2: JetBrains IDE mit DevContainers nutzen
    - Option 3: Docker verwenden:
      `docker build --tag langchain_agents . && docker run -it --rm -v ${PWD}:/workspace -p 8888:8888 langchain_agents`

3. `.env.dist` nach `.env` kopieren und API-Schlüssel konfigurieren:
    - OpenAI API-Schlüssel
    - Tavily und Serparpi Schlüssel für Suchfunktionen

4. **Umgebung testen:** Die Einrichtung sollte vorab getestet werden,
   insbesondere:
    - Notebook-Server erfolgreich starten und auf alle Notebooks zugreifen
      können
    - Einen einfachen API-Call mit OpenAI ausführen, um die API-Schlüssel zu
      validieren
    - Installation aller Abhängigkeiten überprüfen mit
      `pip list | grep langchain`

## Agenda

### Teil 1: Grundlagen (60 Minuten)

1. **Structured Output** (15 Min)
    - Einführung in strukturierte Ausgaben von LLMs
    - Verwendung von ChatPromptTemplate für strukturierte Ausgaben
    - Arbeiten mit JSON, XML und anderen strukturierten Formaten
    - _Notebook_: `01_basics.ipynb`
    - _Code-Beispiel_:
      ```python
      from langchain.prompts import ChatPromptTemplate
      
      prompt = ChatPromptTemplate.from_messages([
          ("system", "Du bist eine hilfsbereicher {beruf} aus Würzburg."),
          ("human", 
           "Erkläre in 2 Sätzen im lokalen Dialekt warum Deine Kunden aus {ort} die besten sind."),
      ])
      # Beispiel für strukturierte Ausgabe mit Format-Anweisungen
      sentiment_prompt = hub.pull("borislove/customer-sentiment-analysis")
      ```
    - _Praxisübung_: Anpassung eines Sentiment-Analyse-Prompts

2. **Tool Calls** (15 Min)
    - Was sind Tool Calls und wie werden sie in LangChain implementiert
    - Erstellen eigener Tools mit dem `@tool` Decorator
    - Integration in einen Agenten
    - _Notebook_: `simple_tool_calling_graph.ipynb`
    - _Code-Beispiel_:
      ```python
      from langchain.tools import tool
      
      @tool
      def test_tool(question: str):
          "This tool provides information about Fritz Karuugaa. Give a question as input."
          return "Fritz Karuugaa is a software developer"
      
      # Tools müssen immer als list übergeben werden
      tools = [test_tool]
      
      # Tools an das llm binden
      llm_with_tools = llm.bind_tools(tools)
      ```
    - _Praxisübung_: Implementierung eines eigenen Tools für Währungsumrechnung
      oder Wetter-Informationen

3. **Modell-Typen im Überblick** (30 Min)
    - Überblick verschiedener LLM-Modelle (OpenAI, Anthropic, Mistral, LLama)
    - Embeddings: Grundlagen, Anwendungsfälle und Implementierung
    - Multimodale Modelle: Beispiele für Text, Bild, Audio mit Gemini
    - _Notebooks_: `01_basics.ipynb`, `15_agent_with_vision.ipynb`
    - _Code-Beispiel für Embeddings_:
      ```python
      import tiktoken
      
      encoding = tiktoken.encoding_for_model("gpt-4o")
      tokens = encoding.encode("AI ist eine tolle Sache.")
      print(tokens)
      decoded_tokens = [encoding.decode_single_token_bytes(token).decode("utf-8") for token in tokens]
      for token in decoded_tokens:
          print(token)
      ```
    - _Praxisübung_: Berechnung von Embeddings für verschiedene Sätze und
      Visualisierung der Embedding-Ähnlichkeit

## PAUSE (10 Minuten)

### Teil 2: Architektur & State Management (60 Minuten)

1. **Chaining** (15 Min)
    - Was ist Chaining und wie funktioniert es in LangChain
    - Der Pipe-Operator `|` und LCEL (LangChain Expression Language)
    - Von einfachen Chains zu komplexen Workflows
    - _Notebooks_: `01_basics.ipynb`, `02_langchain_repo.ipynb`,
      `06_langgraph_intro.ipynb`
    - _Code-Beispiel_:
      ```python
      from langchain.schema import StrOutputParser
      
      chain = prompt | llm() | StrOutputParser()
      print(chain.invoke({"beruf": "Winzer", "ort": "Würzburg"}))
      
      # Streaming
      async for chunk in chain.astream({"beruf": "Winzer", "ort": "Würzburg"}):
          print(chunk, end="", flush=True)
      ```
    - _Praxisübung_: Erstellen einer Chain, die einen Text zusammenfasst und
      dann in eine andere Sprache übersetzt

2. **Architekturkonzepte** (25 Min)
    - Agentenbasierte Ansätze: Grundlagen und Vorteile
    - ReAct Pattern: Reasoning and Acting in LLMs
    - Graph-RAG und Hierarchical RAG: Fortgeschrittene RAG-Architekturen
    - _Notebooks_: `04_react.ipynb`, `05_agentic_rag.ipynb`,
      `19_hierarchical_agent_teams.ipynb`
    - _Code-Beispiel für LangGraph_:
      ```python
      from langgraph.graph import MessageGraph
      
      graph_builder = MessageGraph()
      # Wir definieren unser LLM als einzigen Knoten
      graph_builder.add_node("chatbot_node", model)
      # Dieser Knoten wird der Entrypoint
      graph_builder.set_entry_point("chatbot_node")
      # Von diesem Knoten geht es direkt zu "END", dem vordefinierten Terminalknoten
      graph_builder.set_finish_point("chatbot_node")
      
      # Jetzt kompilieren wir
      simple_graph = graph_builder.compile()
      ```
    - _Praxisübung_: Modifikation eines bestehenden ReAct-Agenten um zusätzliche
      Funktionen

3. **State Management** (20 Min)
    - Warum ist State Management in KI-Anwendungen wichtig?
    - Unterschiedliche Speichertypen in LangChain: Buffer, Summary, Vector
      Memory
    - State Management in LangGraph mit Typisierung
    - _Notebooks_: `08_memory.ipynb`, `06_langgraph_intro.ipynb`
    - _Code-Beispiel_:
      ```python
      from langchain.memory import ConversationBufferMemory
      
      memory = ConversationBufferMemory()
      memory.save_context({"input": "Hallo, ich bin Max"}, {"output": "Hallo Max!"})
      memory.save_context({"input": "Was ist mein Name?"}, {"output": "Dein Name ist Max."})
      
      print(memory.load_memory_variables({}))
      ```
    - _Praxisübung_: Implementierung einer einfachen Chat-Anwendung mit
      unterschiedlichen Memory-Typen

## PAUSE (30 Minuten)

### Teil 3: Datenverwaltung & Infrastruktur (60 Minuten)

1. **Vektordatenbanken** (30 Min)
    - Grundlagen von Vektordatenbanken
    - Wie funktionieren Embeddings und Vektorähnlichkeit
    - Verschiedene Vektordatenbanken im Vergleich
    - Implementierung mit Chroma oder Qdrant
    - Vektorfluss-Optimierung und Query Augmentation
    - RAG-Implementierungsstrategien
    - _Notebook_: `03_rag_basics.ipynb`
    - _Code-Beispiel_:
      ```python
      # Erstellung von Embeddings und Speicherung in einer Vektordatenbank
      from langchain_community.vectorstores import Chroma
      from langchain_community.document_loaders import TextLoader
      
      # Laden von Dokumenten
      loader = TextLoader("state_of_the_union.txt")
      documents = loader.load()
      
      # Erstellen der Vektordatenbank
      db = Chroma.from_documents(documents, embeddings())
      
      # Suchen nach ähnlichen Dokumenten
      query = "Was sind die wichtigsten Punkte zur Wirtschaft?"
      docs = db.similarity_search(query)
      ```
    - _Praxisübung_: Anpassung der RAG-Pipeline für einen eigenen Datensatz

2. **Model Context Protocol (MCP)** (15 Min)
    - Was ist das Model Context Protocol
    - Wie funktioniert Kontextmanagement in LLMs
    - Strategien zur optimalen Kontextnutzung
    - Integration von MCP in LangChain-Anwendungen
    - _Code-Beispiel_:
      ```python
      # Beispiel für Kontext-Management mit LangChain
      from langchain.memory import ConversationBufferMemory
      from langchain.chains import ConversationChain
      
      # Erstellen eines Memory-Objekts für Kontextmanagement
      memory = ConversationBufferMemory()
      conversation = ConversationChain(
          llm=llm(),
          memory=memory,
          verbose=True
      )
      
      # Erste Nachricht
      conversation.predict(input="Hallo! Ich bin Hans.")
      
      # Zweite Nachricht mit Kontext aus der ersten
      conversation.predict(input="Wie ist mein Name?")
      ```
    - _Praxisübung_: Implementierung einer Chat-Anwendung mit verschiedenen
      Memory-Typen und Vergleich der Ergebnisse

3. **Prompt-Management** (15 Min)
    - Organisation und Versionierung von Prompts
    - LangChain Hub für Prompt-Sharing
    - Integration mit Langfuse für Monitoring und Tracking
    - Best Practices für Prompt-Management in größeren Teams
    - _Notebooks_: `01_basics.ipynb`, `07_checkpoints.ipynb`
    - _Code-Beispiel_:
      ```python
      from langchain import hub
      
      # Laden eines Prompts aus dem Hub
      sentiment_prompt = hub.pull("borislove/customer-sentiment-analysis")
      
      # Anpassen und Verwenden des Prompts
      client_letter = """Ich bin von dem Volleyballschläger zutiefst enttäuscht. [...]"""
      format_instructions = """Zusätzlich zur numerischen Klassifizierung sollst du [...]"""
      
      print(sentiment_prompt.format(client_letter=client_letter, 
                                    format_instructions=format_instructions))
      ```
    - _Praxisübung_: Eigenen Prompt im LangChain Hub veröffentlichen

## PAUSE (10 Minuten)

### Teil 4: Tools & Frameworks (50 Minuten)

1. **LangGraph** (25 Min)
    - Funktionsweise und Architektur von LangGraph
    - Unterschied zwischen LangChain und LangGraph
    - Erstellung von Agenten mit mehreren Werkzeugen
    - Praktische Anwendungsbeispiele mit verschiedenen Tools
    - Integration in bestehende Systeme
    - _Notebooks_: `06_langgraph_intro.ipynb`, `simple_tool_calling_graph.ipynb`
    - _Code-Beispiel_:
      ```python
      from langgraph.graph import StateGraph
      from typing import Annotated, TypedDict
      import operator
      
      
      # State Definition für den Graphen
      class AgentState(TypedDict):
          messages: Annotated[list[BaseMessage], operator.add]
      
      
      # Graph aufbauen
      workflow = StateGraph(AgentState)
      workflow.add_node("agent", call_model)
      workflow.add_node("action", call_tools)
      workflow.set_entry_point("agent")
      
      # Bedingungen für Verzweigungen hinzufügen
      workflow.add_conditional_edges(
          "agent",
          should_continue,
          {
              "continue": "action",
              "end": END,
          },
      )
      workflow.add_edge("action", "agent")
      
      # Graph kompilieren
      graph = workflow.compile()
      ```
    - _Praxisübung_: Erweiterung eines LangGraph-Agenten um zusätzliche Tools
      und Entscheidungslogik

2. **LangFuse** (25 Min)
    - Einführung in LangFuse: Monitoring und Tracing von LLM-Anwendungen
    - Installation und Konfiguration von LangFuse
    - Integration in LangChain und LangGraph
    - Tracing von Anfragen und Antworten
    - Analyse von Kosten, Latenz und Qualität
    - Nutzungsmöglichkeiten für Optimierung
    - _Code-Beispiel_:
      ```python
      # Installation: pip install langfuse langchain-langfuse
      
      import os
      from langfuse import Langfuse
      from langfuse.client import StatelessTracer
      
      # LangFuse initialisieren
      langfuse = Langfuse(
          public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
          secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
          host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
      )
      
      # Tracing einer einfachen Konversation
      tracer = StatelessTracer(
          public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
          secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
          host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
      )
      
      # Einfaches Tracing mit LangFuse
      trace = tracer.trace(name="simple_conversation")
      generation = trace.generation(
          name="initial_response",
          model="gpt-4",
          prompt="Was ist KI?",
          completion="KI steht für künstliche Intelligenz..."
      )
      ```
    - _Praxisübung_: Integration von LangFuse in eine einfache
      LangChain-Anwendung und Analyse der Ergebnisse

## PAUSE (10 Minuten)

### Teil 5: Qualitätssicherung & Evaluierung (30 Minuten)

1. **Evaluationsmethoden** (20 Min)
    - LLM-As-A-Judge: Grundlagen und Implementierung
    - NLP-basierte Testkriterien (ROUGE, BLEU, etc.)
    - PI Scrubbing und Datenschutz mit Microsoft Presidio
    - Integration von Evaluierungsmethoden in den Entwicklungsworkflow
    - Praktische Beispiele für verschiedene Anwendungsfälle
    - _Notebooks_: `27_evaluationsmethoden.ipynb`, `22_evaluation.ipynb`,
      `23_Rag_eval.ipynb`
    - _Code-Beispiel für LLM-As-A-Judge_:
      ```python
      from langchain.evaluation import load_evaluator
      
      # Evaluator für Frage-Antwort-Qualität laden
      evaluator = load_evaluator("qa")
      
      # Bewertung einer Antwort
      eval_result = evaluator.evaluate_strings(
          prediction="Berlin ist die Hauptstadt von Deutschland und hat etwa 3,7 Millionen Einwohner.",
          reference="Berlin ist die Hauptstadt Deutschlands.",
          input="Was ist die Hauptstadt von Deutschland?"
      )
      
      print(f"Bewertung: {eval_result}")
      ```
    - _Beispiel für PI Scrubbing mit Presidio_:
      ```python
      from presidio_analyzer import AnalyzerEngine
      from presidio_anonymizer import AnonymizerEngine
      
      # Beispieltext mit personenbezogenen Daten
      text_mit_pii = """
      Sehr geehrter Herr Müller,
      
      Bitte kontaktieren Sie mich unter meiner E-Mail max.schmidt@example.com oder telefonisch unter +49 176 12345678.
      
      Ihr Kundenkonto mit der Nummer DE987654321 wurde aktualisiert.
      
      Mit freundlichen Grüßen,
      Dr. Anna Weber
      """
      
      # Analyzer und Anonymizer initialisieren
      analyzer = AnalyzerEngine()
      anonymizer = AnonymizerEngine()
      
      # Text analysieren und PII erkennen (deutsche Sprache)
      results = analyzer.analyze(text=text_mit_pii, language="de")
      
      # Text anonymisieren
      anonymized_text = anonymizer.anonymize(
          text=text_mit_pii,
          analyzer_results=results
      )
      
      print("Anonymisierter Text:")
      print(anonymized_text.text)
      ```
    - _Praxisübung_: Bewertung verschiedener LLM-Antworten mit dem
      LLM-As-A-Judge-Ansatz und Anonymisierung eines eigenen Beispieltextes

2. **Abschluss und Fragen** (10 Min)
    - Zusammenfassung der wichtigsten Konzepte des Workshops
    - Diskussion über praktische Anwendungsmöglichkeiten
    - Empfehlungen für weiterführende Ressourcen und Lernmaterialien
    - Offene Fragerunde für spezifische Anwendungsfälle
    - Abschlussfeedback und nächste Schritte

## Zusätzliche Ressourcen

- [LangChain Dokumentation](https://python.langchain.com/docs/)
- [LangGraph Dokumentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/) für Monitoring und Debugging
- [LangChain Hub](https://smith.langchain.com/hub) für Prompts und Templates
- [LangFuse](https://langfuse.com/) für Observability
- [RAGAS Dokumentation](https://docs.ragas.io/) für RAG-Evaluation
- [Microsoft Presidio](https://microsoft.github.io/presidio/) für PII-Erkennung
  und Anonymisierung
-
GitHub-Repository: [langchain_agents](https://github.com/mayflower/langchain_agents)

## Nach dem Workshop

- Bereitstellung aller Code-Beispiele und Lösungen
- Follow-up E-Mail mit zusätzlichen Ressourcen
- Optional: Angebot eines Follow-up-Meetings für Fragen, die nach dem Workshop
  aufkommen

## Hinweise für den Dozenten

- Für jeden Abschnitt sind konkrete Jupyter Notebooks zugeordnet, die als
  Demonstrations- und Übungsbasis dienen
- Die Praxisübungen sollten interaktiv gestaltet werden und den Teilnehmern Zeit
  zum Experimentieren geben
- Bei der Installation helfen und sicherstellen, dass alle API-Schlüssel
  funktionieren
- Berücksichtigen Sie unterschiedliche Vorkenntnisse der Teilnehmer und bieten
  Sie Hilfestellungen an
- Legen Sie Wert auf praktische Anwendungsfälle und zeigen Sie konkrete
  Beispiele aus der Praxis
- Stellen Sie sicher, dass alle beispielhaften Codeblöcke funktionieren und gut
  dokumentiert sind
- Bereiten Sie Fallback-Szenarien vor für den Fall, dass API-Dienste während des
  Workshops nicht verfügbar sind
