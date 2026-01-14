"""Basic API health tests"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint returns consistent APIResponse format."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Verify consistent APIResponse structure
    assert data["success"] is True
    assert "data" in data
    assert "meta" in data

    # Verify health data
    assert data["data"]["status"] == "healthy"
    assert "version" in data["data"]
    assert "model_loaded" in data["data"]

    # Verify meta has timestamp
    assert "timestamp" in data["meta"]
