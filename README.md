# 🦜🔗 Autonome Agenten mit LangChain
Dies ist das Repository für unseren Workshop. Die Folien findest du (hier)[https://slides.com/johann-peterhartmann/autonome-agenten/].

# Setup

## Voraussetzungen:
Der einfachste Weg, um loszulegen, ist die Verwendung von VSCode und Devcontainers. Dafür müssen Docker und VSCode auf deinem Rechner installiert sein. Du kannst auch das Dockerfile oder eine lokale Python-Umgebung (falls du Docker aus religiösen Gründen nicht magst) verwenden. Der Devcontainer bietet lediglich eine einheitliche Umgebung für alle.

## Get the source, luke!

```bash
git clone https://github.com/mayflower/langchain_agents.git
cd langchain_agents
cp .env.dist .env
```

# Get the keys!


## OpenAI

(Klick)[]

Das Passwort bekommst du von uns.


## Keys für Tavily und Serparpi

Hier registrieren [SerpApi](https://serpapi.com/).
Und einen key erstellen: [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key)
Zum .env file hinzufügen

Für Tavily [hier](klicken). Ebenfalls einen key erstellen und zu den .env hinzufügen.

## Wer im Dev-Container unterwegs sein will

Unten links in VSCode auf das blaue Viereck klicken. "Reopen in Container". Wenn er dann fragt, ob er den Container neu bauen soll, einmal bestätigen.

## Für docker
```bash
docker build --tag langchain_agents .
docker run -it --rm -v  ${PWD}:/workspace -p 8888:8888 langchain_agents
```
# Go!
