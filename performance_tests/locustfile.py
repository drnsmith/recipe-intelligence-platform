from locust import HttpUser, task, between

class RecipeApiUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def recommend(self):
        self.client.post("/recommend", json={
            "ingredients": "tomato",
            "preferences": ["vegan", "gluten-free"],
            "top_n": 5
        })

    @task
    def substitute(self):
        self.client.post("/substitute", json={"ingredient": "sugar"})
