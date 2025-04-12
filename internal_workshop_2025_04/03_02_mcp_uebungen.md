# Übungen: Model Context Protocol (MCP)

## Übung 1: Vergleich verschiedener Memory-Typen

In dieser Übung vergleichen Sie verschiedene Memory-Typen in LangChain und analysieren ihre Vor- und Nachteile.

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import time

# LLM initialisieren
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Verschiedene Memory-Typen erstellen
buffer_memory = ConversationBufferMemory()
summary_memory = ConversationSummaryMemory(llm=llm)
window_memory = ConversationBufferWindowMemory(k=2)  # Nur die letzten 2 Nachrichten

# Konversationsketten mit unterschiedlichen Memory-Typen erstellen
buffer_chain = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

summary_chain = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

window_chain = ConversationChain(
    llm=llm,
    memory=window_memory,
    verbose=True
)

# Durchführen einer Konversation mit mehreren Nachrichtenaustauschen
conversation_inputs = [
    "Mein Name ist Julia",
    "Ich interessiere mich für künstliche Intelligenz",
    "Speziell für Natural Language Processing",
    "Ich arbeite an einem Projekt zur Textzusammenfassung",
    "Kennst du gute Ressourcen zu diesem Thema?",
    "Wie heißt die Person, mit der du sprichst, und wofür interessiert sie sich?"
]

# Funktionen zum Messen der Antwortzeit und Token-Anzahl
def run_conversation(chain, inputs):
    responses = []
    times = []

    for input_text in inputs:
        start_time = time.time()
        response = chain.predict(input=input_text)
        end_time = time.time()

        responses.append(response)
        times.append(end_time - start_time)

    return responses, times

# Durchführen der Konversationen mit verschiedenen Memory-Typen
buffer_responses, buffer_times = run_conversation(buffer_chain, conversation_inputs)
summary_responses, summary_times = run_conversation(summary_chain, conversation_inputs)
window_responses, window_times = run_conversation(window_chain, conversation_inputs)

# Ergebnisse anzeigen
print("Buffer Memory - Letzte Antwort:")
print(buffer_responses[-1])
print(f"Durchschnittliche Antwortzeit: {sum(buffer_times)/len(buffer_times):.2f} Sekunden\n")

print("Summary Memory - Letzte Antwort:")
print(summary_responses[-1])
print(f"Durchschnittliche Antwortzeit: {sum(summary_times)/len(summary_times):.2f} Sekunden\n")

print("Window Memory - Letzte Antwort:")
print(window_responses[-1])
print(f"Durchschnittliche Antwortzeit: {sum(window_times)/len(window_times):.2f} Sekunden\n")

# Vergleich der Memory-Inhalte
print("Buffer Memory Inhalt:")
print(buffer_memory.load_memory_variables({}))

print("Summary Memory Inhalt:")
print(summary_memory.load_memory_variables({}))

print("Window Memory Inhalt:")
print(window_memory.load_memory_variables({}))
```

**Aufgaben:**
1. Führen Sie den obigen Code aus und vergleichen Sie die Ergebnisse
2. Analysieren Sie die Unterschiede in den Antworten der letzten Frage
3. Vergleichen Sie die Antwortzeiten der verschiedenen Memory-Typen
4. Untersuchen Sie die gespeicherten Memory-Inhalte und bewerten Sie deren Umfang

**Fragen zur Diskussion:**
- Welcher Memory-Typ eignet sich am besten für langfristige Konversationen?
- Welcher Memory-Typ bietet die beste Balance zwischen Kontext und Effizienz?
- In welchen Szenarien würden Sie welchen Memory-Typ empfehlen?

## Übung 2: Implementierung eines eigenen Entity Memory

In dieser Übung implementieren Sie ein spezialisiertes Memory, das wichtige Informationen über Entitäten speichert.

```python
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import json

# LLM initialisieren
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Entity Memory erstellen
entity_memory = ConversationEntityMemory(llm=llm)

# Konversationskette mit Entity Memory erstellen
conversation = ConversationChain(
    llm=llm,
    memory=entity_memory,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    verbose=True
)

# Eine Konversation mit mehreren Entitäten führen
conversation_steps = [
    "Mein Name ist Thomas Schmidt und ich arbeite bei der Firma TechSolutions.",
    "TechSolutions ist ein Unternehmen im Bereich Künstliche Intelligenz mit Sitz in München.",
    "Mein Kollege Michael und ich arbeiten an einem Projekt namens 'SmartAssist'.",
    "Das Projekt SmartAssist verwendet LLMs und soll Unternehmen bei der Kundenbetreuung unterstützen.",
    "Unser Konkurrent ist die Firma AI Dynamics aus Berlin.",
    "Was weißt du über mich, TechSolutions und unser Projekt?"
]

# Konversation durchführen
for step in conversation_steps:
    response = conversation.predict(input=step)
    print(f"Human: {step}")
    print(f"AI: {response}\n")

# Entity Memory anzeigen
entities = entity_memory.entity_store.store
print("Gespeicherte Entitäten:")
for entity, info in entities.items():
    print(f"\nEntität: {entity}")
    print(f"Information: {info}")

# Entity Memory in JSON-Format exportieren
entities_json = json.dumps(entities, indent=2)
print("\nEntity Memory als JSON:")
print(entities_json)
```

**Aufgaben:**
1. Führen Sie den Code aus und analysieren Sie, welche Entitäten erkannt wurden
2. Erweitern Sie die Konversation um mindestens 3 weitere Nachrichten, die neue Entitäten einführen
3. Erstellen Sie eine Funktion, die eine spezifische Entität abfragt und nur Informationen zu dieser zurückgibt
4. Implementieren Sie eine Funktion, die das Entity Memory in eine Datei speichert und später wieder laden kann

**Erweiterung:**
Implementieren Sie ein erweitertes Entity Memory, das auch Beziehungen zwischen Entitäten speichert.

## Übung 3: Tokenverbrauch und Kontextoptimierung

In dieser Übung analysieren Sie den Tokenverbrauch verschiedener Kontextmanagement-Strategien und optimieren eine Konversation für effiziente Tokennutzung.

```python
import tiktoken
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# Funktion zur Zählung von Tokens
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# LLM initialisieren
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Memory-Objekte erstellen
buffer_memory = ConversationBufferMemory()
summary_memory = ConversationSummaryMemory(llm=llm)

# Beispielkonversation
conversation_steps = [
    "Ich möchte mehr über die Geschichte der Programmiersprachen erfahren.",
    "Welche war die erste Hochsprache?",
    "Wann wurde Python entwickelt?",
    "Welche Programmiersprachen sind heute am beliebtesten?",
    "Wie unterscheiden sich funktionale und objektorientierte Programmierung?",
    "Was sind die Vorteile von statisch typisierten Sprachen?",
    "Erkläre mir den Unterschied zwischen Kompilierung und Interpretation.",
    "Welche Programmiersprache würdest du für Webentwicklung empfehlen?"
]

# Konversation mit beiden Memory-Typen durchführen
for step in conversation_steps:
    buffer_memory.save_context({"input": step}, {"output": "Eine sehr ausführliche Antwort über " + step})
    summary_memory.save_context({"input": step}, {"output": "Eine sehr ausführliche Antwort über " + step})

# Tokens in beiden Memory-Typen zählen
buffer_content = buffer_memory.load_memory_variables({})["history"]
summary_content = summary_memory.load_memory_variables({})["history"]

buffer_tokens = count_tokens(buffer_content)
summary_tokens = count_tokens(summary_content)

print(f"Buffer Memory enthält {buffer_tokens} Tokens")
print(f"Summary Memory enthält {summary_tokens} Tokens")
print(f"Tokenreduktion durch Summary Memory: {(1 - summary_tokens/buffer_tokens)*100:.2f}%\n")

# Implementieren Sie eine eigene Token-Optimierungsstrategie
def optimize_context(conversation_history, max_tokens=1000):
    current_tokens = count_tokens(conversation_history)

    if current_tokens <= max_tokens:
        return conversation_history

    # Hier Ihre Optimierungsstrategie implementieren
    # Beispiel: Ältere Nachrichten entfernen oder zusammenfassen

    lines = conversation_history.split('\n')

    # Strategie 1: Ältere Nachrichten entfernen
    while count_tokens('\n'.join(lines)) > max_tokens and len(lines) > 2:
        # Entferne die älteste Nachricht (2 Zeilen - Frage und Antwort)
        lines = lines[2:]

    return '\n'.join(lines)

# Optimierungsstrategie testen
optimized_buffer = optimize_context(buffer_content, 500)
optimized_tokens = count_tokens(optimized_buffer)

print(f"Optimierter Kontext enthält {optimized_tokens} Tokens")
print(f"Tokenreduktion durch Optimierung: {(1 - optimized_tokens/buffer_tokens)*100:.2f}%")
print("Optimierter Kontext:")
print(optimized_buffer)
```

**Aufgaben:**
1. Führen Sie den Code aus und analysieren Sie die Tokenzahlen
2. Verbessern Sie die `optimize_context` Funktion mit einer der folgenden Strategien:
   - Zusammenfassen älterer Nachrichten mit einem LLM
   - Entfernen unwichtiger Informationen basierend auf einem Relevanzfilter
   - Komprimieren des Textes durch Entfernen redundanter Informationen
3. Testen Sie Ihre optimierte Funktion mit verschiedenen Maximalwerten für Tokens
4. Entwickeln Sie eine Strategie, die wichtige Informationen priorisiert und unwichtige verwirft

**Fortgeschrittene Aufgabe:**
Implementieren Sie ein hybrides Memory-System, das Buffer Memory für aktuelle Nachrichten und Summary Memory für ältere Nachrichten verwendet.

## Übung 4: Implementierung eines Hierarchischen Gedächtnisses

In dieser Übung erstellen Sie ein hierarchisches Gedächtnissystem, das verschiedene Memory-Typen kombiniert.

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, List, Tuple
import time

class HierarchicalMemory:
    def __init__(self, llm):
        self.llm = llm
        self.short_term_memory = ConversationBufferWindowMemory(k=5)  # Letzte 5 Nachrichten
        self.long_term_memory = ConversationSummaryMemory(llm=llm)
        self.important_facts = []

        # Template für die Extraktion wichtiger Fakten
        self.fact_extraction_template = PromptTemplate(
            input_variables=["conversation"],
            template="""
            Aus folgender Konversation extrahiere die 3 wichtigsten Fakten,
            die für zukünftige Gespräche relevant sein könnten:

            {conversation}

            Wichtige Fakten (Gebe nur die Fakten zurück, einen pro Zeile):
            """
        )

        self.fact_extraction_chain = LLMChain(
            llm=llm,
            prompt=self.fact_extraction_template
        )

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        # Kontext in beide Memory-Typen speichern
        self.short_term_memory.save_context(inputs, outputs)
        self.long_term_memory.save_context(inputs, outputs)

        # Wichtige Fakten extrahieren (jede 3. Nachricht)
        if len(self.important_facts) % 3 == 0:
            conversation = self.short_term_memory.load_memory_variables({})["history"]
            facts = self.fact_extraction_chain.run(conversation=conversation)

            # Neue Fakten zur Liste hinzufügen
            for fact in facts.strip().split('\n'):
                if fact and fact not in self.important_facts:
                    self.important_facts.append(fact)

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Kurzzeitgedächtnis laden
        short_term = self.short_term_memory.load_memory_variables({})

        # Langzeitgedächtnis laden
        long_term = self.long_term_memory.load_memory_variables({})

        # Wichtige Fakten formatieren
        facts_text = "\n".join([f"- {fact}" for fact in self.important_facts])

        # Kombiniertes Gedächtnis erstellen
        return {
            "short_term_history": short_term["history"],
            "long_term_history": long_term["history"],
            "important_facts": facts_text,
            "combined_history": f"""
            Aktuelle Konversation:
            {short_term["history"]}

            Zusammenfassung früherer Gespräche:
            {long_term["history"]}

            Wichtige Fakten:
            {facts_text}
            """
        }

# Test des hierarchischen Gedächtnisses
llm = ChatOpenAI(model="gpt-3.5-turbo")
hierarchical_memory = HierarchicalMemory(llm)

# Beispielkonversation
conversation_steps = [
    ("Hallo, mein Name ist Markus", "Hallo Markus, wie kann ich dir helfen?"),
    ("Ich arbeite an einem Projekt zur Datenvisualisierung", "Das klingt interessant. Welche Art von Daten visualisierst du?"),
    ("Hauptsächlich Finanzdaten von Unternehmen", "Finanzdaten sind ein gutes Anwendungsgebiet für Visualisierungen."),
    ("Ich verwende Python mit Matplotlib und Plotly", "Das sind ausgezeichnete Bibliotheken für Datenvisualisierung in Python."),
    ("Mein Problem ist die Darstellung von Zeitreihen", "Zeitreihen können komplex sein. Hast du schon spezielle Zeitreihen-Visualisierungen probiert?"),
    ("Welche Bibliothek würdest du für interaktive Finanz-Dashboards empfehlen?", "Für interaktive Finanz-Dashboards empfehle ich Dash von Plotly oder Streamlit.")
]

# Konversation simulieren
for user_input, ai_response in conversation_steps:
    hierarchical_memory.save_context({"input": user_input}, {"output": ai_response})
    time.sleep(1)  # Pause zwischen den Nachrichten

# Gedächtnis abrufen und anzeigen
memory_vars = hierarchical_memory.load_memory_variables({})

print("Kurzzeit-Gedächtnis:")
print(memory_vars["short_term_history"])

print("\nLangzeit-Gedächtnis:")
print(memory_vars["long_term_history"])

print("\nWichtige Fakten:")
print(memory_vars["important_facts"])

print("\nKombiniertes Gedächtnis:")
print(memory_vars["combined_history"])
```

**Aufgaben:**
1. Führen Sie den Code aus und analysieren Sie die verschiedenen Arten von gespeicherten Informationen
2. Erweitern Sie die Klasse `HierarchicalMemory` um eine Methode zum Abfragen bestimmter Informationen
3. Fügen Sie eine Funktion hinzu, die den Speicherverbrauch (Tokens) für jede Gedächtniskomponente berechnet
4. Implementieren Sie eine Strategie zum "Vergessen" unwichtiger Informationen im Langzeitgedächtnis
5. Erweitern Sie die Konversation um weitere Nachrichten und beobachten Sie, wie sich das hierarchische Gedächtnis entwickelt

**Fortgeschrittene Aufgabe:**
Integrieren Sie das hierarchische Gedächtnis in einen Agenten und testen Sie es in einer interaktiven Konversation.

## Übung 5: Kontext-gesteuerte Antwortgenerierung

In dieser Übung implementieren Sie ein System, das den Kontext dynamisch anpasst, um fokussierte Antworten zu generieren.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
import json

# LLM initialisieren
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Memory initialisieren
memory = ConversationBufferMemory(return_messages=True)

# Kontext-Extraktor erstellen
context_analyzer_prompt = ChatPromptTemplate.from_template(
    """
    Basierend auf der folgenden Konversation und der aktuellen Frage,
    identifiziere die relevantesten Informationen für die Beantwortung der Frage.

    Konversationshistorie:
    {history}

    Aktuelle Frage: {question}

    Bitte gib die relevanten Informationen als JSON zurück mit folgenden Feldern:
    - "relevant_topics": Liste der relevanten Themen
    - "key_entities": Liste der wichtigsten Entitäten
    - "context_importance": Eine Bewertung der Wichtigkeit des Kontexts (1-10)

    Antwort (nur JSON):
    """
)

context_analyzer_chain = context_analyzer_prompt | llm | StrOutputParser()

# Antwort-Generator erstellen
answer_prompt = ChatPromptTemplate.from_template(
    """
    Du bist ein hilfreicher Assistent, der präzise und kontextbezogene Antworten gibt.

    Konversationshistorie:
    {history}

    Kontext-Analyse:
    Relevante Themen: {relevant_topics}
    Wichtige Entitäten: {key_entities}
    Kontextrelevanz: {context_importance}/10

    Aktuelle Frage: {question}

    Wenn die Kontextrelevanz über 7 liegt, beziehe dich stark auf den Kontext.
    Bei niedrigerer Relevanz fokussiere dich mehr auf die aktuelle Frage.

    Gib eine präzise und hilfreiche Antwort:
    """
)

answer_chain = answer_prompt | llm | StrOutputParser()

# Gesamtkette erstellen
def context_aware_conversation(question):
    # Konversationshistorie abrufen
    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("messages", [])
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])

    # Kontext analysieren
    context_analysis = context_analyzer_chain.invoke({"history": history_str, "question": question})

    try:
        context_data = json.loads(context_analysis)
    except json.JSONDecodeError:
        # Fallback für den Fall, dass kein gültiges JSON zurückgegeben wird
        context_data = {
            "relevant_topics": ["allgemeine Information"],
            "key_entities": ["keine spezifischen Entitäten"],
            "context_importance": 3
        }

    # Antwort generieren
    answer = answer_chain.invoke({
        "history": history_str,
        "question": question,
        "relevant_topics": ", ".join(context_data["relevant_topics"]),
        "key_entities": ", ".join(context_data["key_entities"]),
        "context_importance": context_data["context_importance"]
    })

    # Konversation in Memory speichern
    memory.save_context({"input": question}, {"output": answer})

    return {
        "answer": answer,
        "context_analysis": context_data
    }

# Funktion zum Testen der Konversation
def run_conversation(questions):
    results = []

    for question in questions:
        print(f"\nFrage: {question}")
        result = context_aware_conversation(question)
        print(f"Kontext-Analyse: {result['context_analysis']}")
        print(f"Antwort: {result['answer']}")
        results.append(result)

    return results

# Beispiel-Konversation
test_questions = [
    "Was sind die besten Programmiersprachen für Data Science?",
    "Welche Bibliotheken werden für maschinelles Lernen verwendet?",
    "Wie funktionieren neuronale Netze?",
    "Was ist deine Lieblingsfarbe?",  # Kontextwechsel
    "Welche Deep Learning Frameworks gibt es?",  # Zurück zum vorherigen Thema
    "Wie kann ich TensorFlow für Bildklassifikation verwenden?"
]

# Konversation durchführen
results = run_conversation(test_questions)
```

**Aufgaben:**
1. Führen Sie den Code aus und analysieren Sie, wie die Kontext-Analyse die Antworten beeinflusst
2. Beobachten Sie, wie das System auf Themenwechsel reagiert
3. Erweitern Sie den Kontext-Analyzer um weitere Metriken, wie z.B.:
   - Sentiment der Konversation
   - Komplexität der Frage
   - Ähnlichkeit zur vorherigen Konversation
4. Implementieren Sie eine Funktion, die automatisch unwichtige Teile des Kontexts entfernt

**Fortgeschrittene Aufgabe:**
Implementieren Sie ein System, das den Kontext in mehrere Ebenen unterteilt (z.B. unmittelbarer Kontext, themenbezogener Kontext, allgemeiner Kontext) und diese dynamisch gewichtet.
