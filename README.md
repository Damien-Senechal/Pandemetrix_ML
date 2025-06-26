# Setup

1. Télécharger et mettre dans data/raw
Cases and Deaths : cases_deaths.csv
Vaccinations : vaccinations_global.csv
Hospitalizations : hospital.csv
Testing : testing.csv

2. Lancer le docker compose dans Pandemetrix_ML
```
docker compose up -d
```
3. Entrer dans le container et s'assurer d'etre dans le dossier app

4. Lancer les commandes :
```
python -c "from app.models import main; main()"

python main.py
```

5. Dans votre navigateur : http://localhost:5001/

