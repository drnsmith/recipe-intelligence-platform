from locust import HttpUser, task, between

class RecipeAPILoadTest(HttpUser):
    host = "http://localhost:8000"  # Set the base URL for your API
    wait_time = between(1, 3)  # Wait between 1-3 seconds between tasks

    @task
    def recommend(self):
        self.client.post("/recommend", json={
            "ingredients": "tomato",
            "preferences": ["cheese", "basil"],
            "top_n": 2
        })

    @task
    def recommend_by_embedding(self):
        self.client.post("/recommend_by_embedding", json={
            "ingredients": "tomato",
            "top_n": 2
        })

    @task
    def substitute_ingredients(self):
        self.client.post("/substitute", json={
            "ingredient": "sugar"
        })
