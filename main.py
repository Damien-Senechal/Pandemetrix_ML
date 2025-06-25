# main.py (REMPLACE tout le contenu)
from app import create_app
from app.routes import load_model_and_metadata

if __name__ == '__main__':
    print("Démarrage de Pandemetrix ML API...")
    
    # Charger le modèle
    model_loaded = load_model_and_metadata()
    
    if not model_loaded:
        print("Modèle non chargé - entraîne d'abord ton modèle:")
        print("python -c \"from app.models import main; main()\"")
    
    # Créer et lancer l'app
    app = create_app()
    
    print("=" * 50)
    print("PANDEMETRIX ML API PRÊTE")
    print("Link : http://localhost:5000")
    print("Modèle chargé:", "OK" if model_loaded else "KO")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
