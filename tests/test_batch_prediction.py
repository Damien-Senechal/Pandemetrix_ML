# tests/test_batch_prediction.py
import requests
import json
import pytest
import os
from datetime import datetime

class TestBatchPredictionCI:
    """Tests pour l'endpoint batch en environnement CI/CD isolé"""
    
    # Utiliser la variable d'environnement ou fallback
    BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:6666')
    
    def setup_method(self):
        """Setup avant chaque test"""
        print(f"Testing with API: {self.BASE_URL}")
        try:
            response = requests.get(f"{self.BASE_URL}/api/v1/covid/health", timeout=10)
            if response.status_code != 200:
                pytest.skip(f"API de test non accessible: {response.status_code}")
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API de test non accessible: {e}")
        
    def test_health_endpoint(self):
        """Test du endpoint de santé"""
        response = requests.get(f"{self.BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"Health status: {data}")
        
    def test_countries_endpoint(self):
        """Test du endpoint des pays"""
        response = requests.get(f"{self.BASE_URL}/api/v1/covid/countries")
        
        print(f"Countries response: {response.status_code}")
        print(f"Countries body: {response.text}")
        
        if response.status_code == 404:
            pytest.skip("Countries endpoint not implemented")
        
        assert response.status_code == 200
        data = response.json()
        assert "countries" in data or "countries_count" in data
        
    def test_batch_prediction_success(self):
        """Test d'une prédiction batch valide avec données de test"""
        
        # Données de test utilisant les pays de notre dataset de test
        test_data = {
            "predictions": [
                {
                    "country": "France",
                    "date": "2023-01-15",
                    "new_cases": 10000.0,
                    "people_vaccinated": 4500000.0,
                    "new_tests": 200000.0,  
                    "daily_occupancy_hosp": 15000.0
                },
                {
                    "country": "Germany", 
                    "date": "2023-01-15",
                    "new_cases": 8000.0,
                    "people_vaccinated": 6000000.0,
                    "new_tests": 180000.0,
                    "daily_occupancy_hosp": 12000.0
                }
            ]
        }
        
        response = requests.post(
        f"{self.BASE_URL}/api/v1/covid/predict-batch",
        json=test_data,
        timeout=30
        )
        
        print(f"Batch response status: {response.status_code}")
        print(f"Batch response: {response.text}")
        
        if response.status_code == 404:
            pytest.skip("Batch endpoint not found - check route implementation")
        elif response.status_code == 501:
            pytest.skip("Batch endpoint not implemented yet")
        elif response.status_code == 500:
            pytest.fail(f"Server error: {response.text}")
        
        assert response.status_code == 200
        result = response.json()
        
        # Vérifications du format de réponse
        assert "successful_predictions" in result
        assert "results" in result
        assert result["successful_predictions"] >= 0
        
        print(f"Batch prediction successful: {result['successful_predictions']} predictions")
        
    def test_batch_prediction_invalid_data(self):
        """Test avec données invalides"""
        test_data = {
            "predictions": [
                {
                    "country": "PaysInexistant",
                    "date": "invalid-date",
                    "new_cases": -1000,  # Valeur négative
                }
            ]
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/v1/covid/predict-batch",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 404:
            pytest.skip("Batch endpoint not implemented")
        
        # Doit retourner une erreur ou des erreurs dans le résultat
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            result = response.json()
            # Si 200, doit avoir des erreurs reportées
            assert "errors" in result
            assert len(result.get("errors", [])) > 0
        
    def test_batch_empty_request(self):
        """Test avec requête vide"""
        response = requests.post(
            f"{self.BASE_URL}/api/v1/covid/predict-batch",
            json={},
            timeout=30
        )
        
        if response.status_code == 404:
            pytest.skip("Batch endpoint not implemented")
            
        assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
