from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_recommend_with_preprocessed_ingredients():
    response = client.post(
        "/recommend",
        json={
            "ingredients": "tomato",
            "preferences": ["cheese", "basil"],
            "top_n": 2
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 2  # Ensure the response contains at most top_n results
    for recipe in data:
        assert "tomato" in recipe['title'].lower() or "tomato" in " ".join(recipe['ingredients']).lower()
