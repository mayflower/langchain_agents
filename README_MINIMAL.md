## Lokale Installation (ohne Docker-Container)

Gilt insbesondere für Rechner mit ARM Architektur (M1 usw.), aber auch wenn man man den ganzen Ballast der 12 GiB Installation des Containers nicht braucht.

- VSCode NICHT sofort starten. Falls VSCode schon gestartet war und der Container schon läuft: VSCode CMD-SHIFT-P (oder links unten) : Reopen locally, Docker beenden
- Aktuelles Python (3.12.3) muss installiert sein auf dem Rechner. Für Mac: `brew install python`
- `pip3` sollte dann schon installiert sein, falls nicht: `python3 -m pip install --upgrade pip`
- Alles folgende im Projektverzeichnis ausführen
    - Virtualenv (venv) local installieren: ```python3 -m venv .venv```
    - In der aktuellen Shell aktivieren: ```source .venv/bin/activate```
    - Restliche requirements installieren: ```pip3 install -r requirements_minimal_frozen.txt```, das dauert einen moment...
    - die .env.dist kopieren nach .env: ```cp .env.dist .env```
    - Den `OPENAI_API_KEY` einfügen, siehe dazu wieder die [README.md](./README.md) 
- Jetzt VSCode starten, `Reopen in Container` NICHT klicken.
- In VSCode: Sollte gemerkt haben, dass sich da was getan hat. Generell zustimmen, wenn er was vorschlägt, im Zweifel nachdenken oder fragen.
- In VSCOde: Mit `Select Kernel` (rechts oben) auf `Python_Environments...`, dann `.venv ...` setzen


Sollte nun gehen. Das erste Beispiel (`pip install -r requirements.txt --quiet`) logischerweise NICHT ausführen.