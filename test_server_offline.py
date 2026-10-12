"""
Local offline validation test for CypherPunks Roadside Assistant API.
Validates the FastAPI routing, schemas, and structure.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_endpoints():
    print("Testing GET / ...")
    r = client.get("/")
    assert r.status_code == 200
    print("Root endpoint OK:", r.json()["project"])

    print("Testing GET /health ...")
    r = client.get("/health")
    assert r.status_code == 200
    print("Health endpoint OK:", r.json()["status"])

    print("Testing GET /api/info ...")
    r = client.get("/api/info")
    assert r.status_code == 200
    print("Info endpoint OK: Team", r.json()["team"])

    print("\nAll offline tests passed successfully!")

if __name__ == "__main__":
    test_endpoints()
