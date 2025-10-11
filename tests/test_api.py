
from fastapi.testclient import TestClient
from app.main import app

# Create a TestClient instance
client = TestClient(app)

def test_health_check():
    """Tests if the health check endpoint is working."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_endpoint_no_docs():
    """
    Tests the chat endpoint's behavior when no documents have been ingested.
    It should gracefully handle this and say it doesn't know.
    """
    # We are using a fresh, in-memory DB for this test, so nothing is ingested.
    response = client.post("/api/v1/chat", json={"query": "What is the warranty?"})
    
    assert response.status_code == 200
    response_json = response.json()
    assert "I could not find any relevant information" in response_json["answer"]
    assert response_json["sources"] == []