# 🦜🔗 Autonome Agenten mit LangChain
Dies ist das Repository für unseren Workshop. Die Folien findest du [hier](https://slides.com/johann-peterhartmann/autonome-agenten/).

# Setup

## Voraussetzungen:

Der einfachste Weg, um loszulegen, ist die Verwendung von VSCode und Devcontainers. Dafür müssen Docker und VSCode auf deinem Rechner installiert sein. Du kannst auch das Dockerfile oder eine lokale Python-Umgebung (falls du Docker aus religiösen Gründen nicht magst) verwenden. Der Devcontainer bietet lediglich eine einheitliche Umgebung für alle.

## Get the source, luke!

```bash
git clone https://github.com/mayflower/langchain_agents.git
cd langchain_agents
cp .env.dist .env
```

## Get the keys!

### .env erstellen

Erstelle eine Kopie von der Datei env.dist und benenne die Kopie .env

### OpenAI

[Klick](https://pass.mayflower.de/#/send/CFMgK7-0QYurwsdG-tujKQ/IpF1KC1zW_5Gy_cROr2QTA)

Das Passwort bekommst du von uns. Bitte danach den key in die .env übertragen als OPENAI_API_KEY


### Keys für Tavily und Serparpi

Hier registrieren [SerpApi](https://serpapi.com/).
Und einen key erstellen: [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key)
Zum .env file hinzufügen

Für Tavily [hier](klicken). Ebenfalls einen key erstellen und zum .env file hinzufügen.

Beide keys sind bis zu einem gewissen Request-Limit/Monat gratis


## Docker

### Wer im Dev-Container unterwegs sein will

Unten links in VSCode auf das blaue Viereck klicken. "Reopen in Container". Wenn er dann fragt, ob er den Container neu bauen soll, einmal bestätigen.

### Für normales docker
```bash
docker build --tag langchain_agents .
docker run -it --rm -v  ${PWD}:/workspace -p 8888:8888 langchain_agents
```

## Go. Notebooks starten

Wer möchte, kann natürlich gerne direkt in VSCode oder einem Editor der Wahl die Notebooks starten. Die Logs von Docker bieten sonst immer einen Link an, wie man Jupyter Lab im Browser benutzen kann. Üblicherweise `localhost:8888`

## Nixos
To use Jupyter Notebook inside venv and therefore being able to use pip install

```
nix develop .
pip install notebook
jupyter notebook
```
