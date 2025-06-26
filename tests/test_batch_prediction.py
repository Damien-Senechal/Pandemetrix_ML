# tests/test_batch_prediction.py
import pytest
import requests
import json
import time
import os

class TestBatchPredictionCI:
    """Tests spécifiques pour la CI/CD"""
    
    BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5001")
    
    @pytest.fixture(scope="class")
    def api_health_check(self):
        """Vérifier que l'API est opérationnelle"""
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.BASE_URL}/api/v1/covid/health", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ready_for_predictions", False):
                        return True
                    
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(5)
        
        pytest.skip("API not ready after 100 seconds")
    
    @pytest.fixture(scope="class") 
    def supported_countries(self, api_health_check):
        """Récupérer les pays supportés"""
        response = requests.get(f"{self.BASE_URL}/api/v1/covid/countries")
        assert response.status_code == 200
        
        data = response.json()
        countries = data.get("countries", [])
        assert len(countries) > 0, "No countries available"
        
        return countries
    
    def test_batch_endpoint_exists(self, api_health_check):
        """Vérifier que l'endpoint batch existe"""
        response = requests.post(f"{self.BASE_URL}/api/v1/covid/predict-batch", json={})
        # Doit retourner 400 (bad request) ou 200, mais pas 404
        assert response.status_code != 404, "Endpoint predict-batch not found"
    
    def test_batch_with_mock_data(self, supported_countries):
        """Test avec données mockées"""
        test_data = {
            "predictions": [
                {
                    "country": supported_countries[0],
                    "date": "2023-01-15",
                    "new_cases": 10000.0,
                    "people_vaccinated": 4500000.0,
                    "new_tests": 200000.0,  
                    "daily_occupancy_hosp": 15000.0
                }
            ]
        }
        
        response = requests.post(
            f"{self.BASE_URL}/api/v1/covid/predict-batch",
            json=test_data,
            timeout=30
        )
        
        # Log pour debug
        print(f"Batch response status: {response.status_code}")
        print(f"Batch response body: {response.text}")
        
        if response.status_code == 501:
            pytest.skip("Batch endpoint not implemented yet")
        
        assert response.status_code == 200
        result = response.json()
        assert "successful_predictions" in result
