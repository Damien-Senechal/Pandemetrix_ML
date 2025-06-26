# app/swagger_models.py
from flask_restx import fields

def create_swagger_models(api):
    """Définit les modèles Swagger pour la documentation"""
    
    # Modèle pour une requête de prédiction simple (UNIQUEMENT 5 features)
    prediction_input = api.model('PredictionInput', {
        'country': fields.String(
            required=True, 
            description='Nom du pays (exactement comme dans /countries)', 
            example='France'
        ),
        'date': fields.String(
            required=True, 
            description='Date au format YYYY-MM-DD', 
            example='2023-01-15'
        ),
        'new_cases': fields.Float(
            required=True, 
            description='Nouveaux cas COVID-19 du jour', 
            example=1500.0,
            min=0
        ),
        'people_vaccinated': fields.Float(
            required=True, 
            description='Nombre total de personnes vaccinées (au moins 1 dose)', 
            example=5000000,
            min=0
        ),
        'new_tests': fields.Float(
            required=True, 
            description='Nouveaux tests effectués ce jour', 
            example=100000,
            min=0
        ),
        'daily_occupancy_hosp': fields.Float(
            required=True, 
            description='Occupation hospitalière quotidienne liée au COVID', 
            example=2500,
            min=0
        )
    })
    
    # Modèle pour une requête batch (plusieurs prédictions)
    batch_input = api.model('BatchInput', {
        'predictions': fields.List(
            fields.Nested(prediction_input), 
            required=True, 
            description='Liste des prédictions à effectuer (max 10)',
            min_items=1,
            max_items=10
        )
    })
    
    # Modèle de réponse pour une prédiction simple
    prediction_output = api.model('PredictionOutput', {
        'prediction': fields.Float(
            description='Prédiction du nombre de nouveaux décès COVID-19',
            example=45.2
        ),
        'prediction_rounded': fields.Integer(
            description='Prédiction arrondie (nombre entier)',
            example=45
        ),
        'confidence_interval': fields.Raw(
            description='Intervalle de confiance (si disponible)',
            example={"lower": 40.1, "upper": 50.3}
        ),
        'input_data': fields.Raw(
            description='Données d\'entrée utilisées pour la prédiction'
        ),
        'model_info': fields.Raw(
            description='Informations sur le modèle utilisé'
        ),
        'timestamp': fields.DateTime(
            description='Horodatage de la prédiction'
        )
    })
    
    # Modèle de réponse batch
    batch_output = api.model('BatchOutput', {
        'results': fields.List(
            fields.Nested(prediction_output),
            description='Liste des résultats de prédiction'
        ),
        'summary': fields.Raw(
            description='Résumé des prédictions',
            example={
                "total_predictions": 3,
                "average_prediction": 42.5,
                "min_prediction": 30.1,
                "max_prediction": 55.8
            }
        ),
        'processing_time': fields.Float(
            description='Temps de traitement en secondes',
            example=0.15
        )
    })
    
    # Modèle pour les informations sur l'API
    api_info = api.model('ApiInfo', {
        'name': fields.String(
            description='Nom de l\'API', 
            example='COVID-19 Prediction API'
        ),
        'version': fields.String(
            description='Version de l\'API', 
            example='1.0.0'
        ),
        'description': fields.String(
            description='Description de l\'API',
            example='API de prédiction des décès COVID-19 avec IA'
        ),
        'model_loaded': fields.Boolean(
            description='Statut du chargement du modèle',
            example=True
        ),
        'features_required': fields.List(
            fields.String,
            description='Features requises pour les prédictions',
            example=["country", "date", "new_cases", "people_vaccinated", "new_tests", "daily_occupancy_hosp"]
        ),
        'countries_supported': fields.Integer(
            description='Nombre de pays supportés',
            example=44
        ),
        'model_performance': fields.Raw(
            description='Métriques de performance du modèle',
            example={
                "r2_score": 0.876,
                "mae": 31.02,
                "training_date": "2024-01-15"
            }
        )
    })
    
    # Modèle pour la liste des pays
    countries_list = api.model('CountriesList', {
        'countries': fields.List(
            fields.String,
            description='Liste des pays supportés par le modèle',
            example=["Algeria", "Argentina", "Australia", "Austria", "Belgium"]
        ),
        'total_countries': fields.Integer(
            description='Nombre total de pays disponibles',
            example=44
        ),
        'sample_countries': fields.List(
            fields.String,
            description='Échantillon des premiers pays',
            example=["Algeria", "Argentina", "Australia", "Austria", "Belgium"]
        ),
        'model_performance': fields.Raw(
            description='Performance du modèle',
            example={
                "r2_score": 0.876,
                "countries_trained": 44
            }
        ),
        'note': fields.String(
            description='Note d\'utilisation',
            example='✅ Utilisez exactement ces noms de pays dans vos requêtes'
        )
    })
    
    # Modèle pour le health check
    health_status = api.model('HealthStatus', {
        'status': fields.String(
            description='Statut général de l\'API',
            example='healthy'
        ),
        'model_loaded': fields.Boolean(
            description='Modèle chargé avec succès',
            example=True
        ),
        'metadata_loaded': fields.Boolean(
            description='Métadonnées chargées',
            example=True
        ),
        'uptime': fields.String(
            description='Temps de fonctionnement',
            example='2 hours'
        ),
        'last_prediction': fields.DateTime(
            description='Dernière prédiction effectuée'
        )
    })
    
    # Modèle pour les erreurs
    error_response = api.model('ErrorResponse', {
        'error': fields.String(
            description='Message d\'erreur',
            example='Pays non supporté'
        ),
        'error_code': fields.String(
            description='Code d\'erreur',
            example='UNSUPPORTED_COUNTRY'
        ),
        'details': fields.Raw(
            description='Détails additionnels sur l\'erreur'
        ),
        'timestamp': fields.DateTime(
            description='Horodatage de l\'erreur'
        )
    })
    
    return {
        'prediction_input': prediction_input,
        'batch_input': batch_input,
        'prediction_output': prediction_output,
        'batch_output': batch_output,
        'api_info': api_info,
        'countries_list': countries_list,
        'health_status': health_status,
        'error_response': error_response
    }
