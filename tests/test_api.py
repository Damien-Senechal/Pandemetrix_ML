# tests/test_api_enhanced.py
import requests
import json
import pytest
from datetime import datetime

BASE_URL = "http://localhost:5000"

class TestAPI:
    """Suite de tests pour l'API COVID-19"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup avant chaque test"""
        # Vérifier que l'API est accessible
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            assert response.status_code == 200, "API non accessible"
        except requests.exceptions.ConnectionError:
            pytest.skip("API non démarrée")
    
    def test_health_endpoint(self):
        """Test du endpoint de santé"""
        response = requests.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
    def test_countries_endpoint(self):
        """Test du endpoint des pays"""
        response = requests.get(f"{BASE_URL}/countries")
        
        assert response.status_code == 200
        data = response.json()
        assert "countries_count" in data
        assert "countries" in data
        assert data["countries_count"] > 0
        assert isinstance(data["countries"], list)
    
    def test_valid_prediction(self):
        """Test d'une prédiction valide"""
        # D'abord récupérer un pays supporté
        countries_resp = requests.get(f"{BASE_URL}/countries")
        countries = countries_resp.json()["countries"]
        test_country = countries[0]  # Premier pays disponible
        
        test_data = {
            "country": test_country,
            "date": "2023-01-15",
            "new_cases": 50000,
            "people_vaccinated": 45000000,
            "new_tests": 200000,
            "daily_occupancy_hosp": 15000
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "new_deaths_predicted" in result["prediction"]
        assert result["prediction"]["new_deaths_predicted"] >= 0
        
    def test_invalid_country_prediction(self):
        """Test avec un pays non supporté"""
        test_data = {
            "country": "PaysInexistant123",
            "date": "2023-01-15",
            "new_cases": 50000,
            "people_vaccinated": 4500000,
            "new_tests": 200000,
            "daily_occupancy_hosp": 15000
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        assert response.status_code == 400
        
    def test_missing_fields_prediction(self):
        """Test avec champs manquants"""
        test_data = {
            "country": "France",
            "date": "2023-01-15"
            # Champs manquants volontairement
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        assert response.status_code == 400

if __name__ == "__main__":
    # Pour les tests manuels
    pytest.main([__file__, "-v"])
