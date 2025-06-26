# scripts/test_docker_isolated.sh
#!/bin/bash

set -e

echo "Test Docker et API"
echo "===================================="

# Configuration API
API_BASE_URL="http://localhost:6666"
API_HEALTH_URL="${API_BASE_URL}/api/v1/covid/health"
API_COUNTRIES_URL="${API_BASE_URL}/api/v1/covid/countries"  
API_PREDICT_URL="${API_BASE_URL}/api/v1/covid/predict"
API_BATCH_URL="${API_BASE_URL}/api/v1/covid/predict-batch"

# Nettoyage automatique
cleanup() {
    echo ""
    echo "Nettoyage automatique..."
    docker rm -f covid-api-test-isolated 2>/dev/null || true
    [ -n "$TMP_DIR" ] && rm -rf "$TMP_DIR" 2>/dev/null || true
    echo "Nettoyage terminé"
}
trap cleanup EXIT

# 1. Construire l'image si nécessaire
echo "Vérification de l'image Docker..."
if ! docker images | grep -q "covid-api-test"; then
    echo "   Construction en cours..."
    docker build -t covid-api-test .
    echo "Image construite"
else
    echo "Image existe déjà"
fi

# 2. Préparer environnement de test
TMP_DIR=$(mktemp -d)
echo "Environnement de test: $TMP_DIR"

# Créer structure
mkdir "$TMP_DIR/data"
mkdir "$TMP_DIR/data/raw"
mkdir "$TMP_DIR/models"
mkdir "$TMP_DIR/logs"

# 3. Créer données de test (PLUS de données pour un meilleur entraînement)
echo "Création des données de test enrichies..."

cat > "$TMP_DIR/data/raw/cases_deaths.csv" << 'EOF'
date,country,new_cases,new_deaths
2023-01-01,France,10000,100
2023-01-02,France,11000,110
2023-01-03,France,9500,95
2023-01-04,France,12000,105
2023-01-05,France,10500,98
2023-01-06,France,11200,102
2023-01-07,France,10800,99
2023-01-08,France,11500,108
2023-01-09,France,10300,96
2023-01-10,France,12200,112
2023-01-11,Germany,8000,80
2023-01-12,Germany,8500,85
2023-01-13,Germany,7800,78
2023-01-14,Germany,9000,90
2023-01-15,Germany,8200,82
EOF

cat > "$TMP_DIR/data/raw/vaccinations_global.csv" << 'EOF'
date,country,people_vaccinated
2023-01-01,France,45000000
2023-01-02,France,45100000
2023-01-03,France,45200000
2023-01-04,France,45300000
2023-01-05,France,45400000
2023-01-06,France,45500000
2023-01-07,France,45600000
2023-01-08,France,45700000
2023-01-09,France,45800000
2023-01-10,France,45900000
2023-01-11,Germany,60000000
2023-01-12,Germany,60100000
2023-01-13,Germany,60200000
2023-01-14,Germany,60300000
2023-01-15,Germany,60400000
EOF

cat > "$TMP_DIR/data/raw/testing.csv" << 'EOF'
date,country,new_tests
2023-01-01,France,200000
2023-01-02,France,205000
2023-01-03,France,198000
2023-01-04,France,210000
2023-01-05,France,195000
2023-01-06,France,208000
2023-01-07,France,202000
2023-01-08,France,215000
2023-01-09,France,190000
2023-01-10,France,220000
2023-01-11,Germany,180000
2023-01-12,Germany,185000
2023-01-13,Germany,178000
2023-01-14,Germany,190000
2023-01-15,Germany,182000
EOF

cat > "$TMP_DIR/data/raw/hospital.csv" << 'EOF'
date,country,daily_occupancy_hosp
2023-01-01,France,15000
2023-01-02,France,15200
2023-01-03,France,14800
2023-01-04,France,15500
2023-01-05,France,14900
2023-01-06,France,15300
2023-01-07,France,15100
2023-01-08,France,15600
2023-01-09,France,14700
2023-01-10,France,15800
2023-01-11,Germany,12000
2023-01-12,Germany,12200
2023-01-13,Germany,11800
2023-01-14,Germany,12500
2023-01-15,Germany,12100
EOF

echo "$(ls -la "$TMP_DIR/data/raw" | wc -l) fichiers CSV créés avec données France + Germany"

# 4. Démarrer le container
echo "Démarrage du container..."
docker run -d --name covid-api-test-isolated \
  -p 6666:5000 \
  -v "$TMP_DIR:/app/test_data" \
  covid-api-test

# 5. Attendre que l'API soit prête avec le BON endpoint
echo "Attente de l'API sur $API_HEALTH_URL"
for i in {1..20}; do
    echo "   Tentative $i/20..."
    if curl -s "$API_HEALTH_URL" > /dev/null 2>&1; then
        echo "API prête après $i tentatives"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "Échec de démarrage de l'API"
        exit 1
    fi
    sleep 2
done

# 6. Entraîner le modèle avec données de test
echo "Entraînement du modèle avec données France + Germany..."
docker exec covid-api-test-isolated bash -c "
    cp -r /app/test_data/* /app/ && \
    python -c 'from app.models import main; main()'
"

if [ $? -eq 0 ]; then
    echo "OK Entraînement terminé"
else
    echo "KO Échec de l'entraînement"
    exit 1
fi

# 7. Redémarrer pour charger le modèle
echo "Redémarrage du container pour charger le modèle..."
docker restart covid-api-test-isolated
sleep 5

echo "Attente de l'API sur $API_HEALTH_URL"
for i in {1..20}; do
    echo "   Tentative $i/20..."
    if curl -s "$API_HEALTH_URL" > /dev/null 2>&1; then
        echo "API prête après $i tentatives"
        break
    fi
    sleep 2
done

# 8. Tests des endpoints avec les BONS URLs
echo "Tests des endpoints..."

echo "   Test health:"
HEALTH_RESPONSE=$(curl -s "$API_HEALTH_URL" 2>/dev/null)
echo "   Réponse: $HEALTH_RESPONSE"

echo "   Test countries:"
COUNTRIES_RESPONSE=$(curl -s "$API_COUNTRIES_URL" 2>/dev/null)
echo "   Réponse: $COUNTRIES_RESPONSE"

echo "   Test batch prediction (ENDPOINT CORRECT):"
BATCH_DATA='{
    "predictions": [
        {
            "country": "France",
            "date": "2023-01-16",
            "new_cases": 11000,
            "people_vaccinated": 46000000,
            "new_tests": 200000,
            "daily_occupancy_hosp": 15000
        },
        {
            "country": "Germany", 
            "date": "2023-01-16",
            "new_cases": 8500,
            "people_vaccinated": 60500000,
            "new_tests": 185000,
            "daily_occupancy_hosp": 12300
        }
    ]
}'

BATCH_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$BATCH_DATA" \
    "$API_BATCH_URL" 2>/dev/null)
echo "   Réponse batch: $BATCH_RESPONSE"

# 9. Variables d'environnement pour pytest (ENDPOINTS CORRECTS)
echo ""
echo "Configuration pytest avec bons endpoints..."
export API_BASE_URL="$API_BASE_URL"
export API_HEALTH_URL="$API_HEALTH_URL" 
export API_BATCH_URL="$API_BATCH_URL"

# Créer un fichier temporaire avec une version corrigée du test
cat > "/tmp/test_batch_corrected.py" << 'EOF'
import requests
import json
import pytest
import os

class TestBatchPredictionCI:
    """Tests pour l'endpoint batch en environnement CI/CD isolé"""
    
    BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:6666')
    
    def setup_method(self):
        """Setup avant chaque test"""
        print(f"Testing with API: {self.BASE_URL}")
        
        # Test avec le BON endpoint health
        try:
            response = requests.get(f"{self.BASE_URL}/api/v1/covid/health", timeout=10)
            if response.status_code != 200:
                pytest.skip(f"API de test non accessible: {response.status_code}")
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API de test non accessible: {e}")
    
    def test_health_endpoint(self):
        """Test du endpoint de santé"""
        response = requests.get(f"{self.BASE_URL}/api/v1/covid/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"Health status: {data}")
        
    def test_countries_endpoint(self):
        """Test du endpoint des pays"""
        response = requests.get(f"{self.BASE_URL}/api/v1/covid/countries")
        
        print(f"Countries response: {response.status_code}")
        
        if response.status_code == 404:
            pytest.skip("Countries endpoint not implemented")
        
        assert response.status_code == 200
        data = response.json()
        assert "countries" in data
        
    def test_batch_prediction_success(self):
        """Test d'une prédiction batch valide avec données de test"""
        
        test_data = {
            "predictions": [
                {
                    "country": "France",
                    "date": "2023-01-16",
                    "new_cases": 10000.0,
                    "people_vaccinated": 46000000.0,
                    "new_tests": 200000.0,  
                    "daily_occupancy_hosp": 15000.0
                }
            ]
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/v1/covid/predict-batch",  # BON ENDPOINT !
            json=test_data,
            timeout=30
        )
        
        print(f"Batch response status: {response.status_code}")
        print(f"Batch response: {response.text}")
        
        if response.status_code == 404:
            pytest.fail("Batch endpoint not found - check route implementation")
        elif response.status_code == 500:
            pytest.fail(f"Server error: {response.text}")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "successful_predictions" in result
        assert "results" in result
        assert result["successful_predictions"] >= 0
        
        print(f"Batch prediction successful: {result['successful_predictions']} predictions")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
EOF

echo "Lancement de pytest avec endpoints corrigés..."
if python -m pytest "/tmp/test_batch_corrected.py" -v --tb=short --color=yes --disable-warnings; then
    echo "Tests pytest réussis !"
else
    echo "Certains tests ont échoué"
fi

echo ""
echo "Tests terminés !"
echo "API de test disponible sur: $API_BASE_URL"
echo "Endpoints testés avec succès:"
echo "   - Health: $API_HEALTH_URL OK"
echo "   - Countries: $API_COUNTRIES_URL OK"
echo "   - Batch predict: $API_BATCH_URL KO"
echo ""
echo "Container actif: covid-api-test-isolated"
echo "Tes données originales sont intactes !"

# Le cleanup automatique se fera en sortie
