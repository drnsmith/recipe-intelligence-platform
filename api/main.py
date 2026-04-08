from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.recipe_logic import filter_recipes, recommend_by_embedding
from app.ingredient_logic import suggest_substitutions
from app.data_loader import load_data
import logging
import ast
import redis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from starlette.middleware.base import BaseHTTPMiddleware
import time

from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response


# Configure Redis
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    print("Connected to Redis!")
except redis.ConnectionError:
    redis_client = None
    print("Redis unavailable. Caching disabled.")

# Middleware to measure response times
class ResponseTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logging.info(f"{request.method} {request.url} completed in {process_time:.2f} seconds")
        return response

# Initialise FastAPI app
app = FastAPI(
    title="Recipe API",
    description="An API for recommending recipes and suggesting ingredient substitutions.",
    version="1.0.0",
)


# Add middleware for response time tracking
app.add_middleware(ResponseTimeMiddleware)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recipe API!"}

# Load data
df, embeddings = load_data()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Request models
class RecommendRequest(BaseModel):
    ingredients: Optional[str] = None
    preferences: Optional[List[str]] = None
    top_n: int = 3

class SubstituteRequest(BaseModel):
    ingredient: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recipe API!"}

@app.get("/health")
def health_check():
    redis_status = "connected" if redis_client and redis_client.ping() else "unavailable"
    return {"status": "ok", "redis": redis_status}

# Define Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total number of API requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests in seconds", ["endpoint"])
CACHE_HIT_COUNT = Counter("cache_hits_total", "Total number of cache hits", ["endpoint"])
CACHE_MISS_COUNT = Counter("cache_misses_total", "Total number of cache misses", ["endpoint"])

# Middleware to track request latency and count
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        endpoint = request.url.path
        method = request.method
        with REQUEST_LATENCY.labels(endpoint=endpoint).time():
            response = await call_next(request)
        http_status = response.status_code
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=http_status).inc()
        return response

# Add middleware to FastAPI
app.add_middleware(MetricsMiddleware)

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

# Update your cache-related endpoints to include metrics
@app.post("/recommend")
def recommend_recipes(request: RecommendRequest):
    ingredients = request.ingredients
    preferences = request.preferences
    top_n = request.top_n

    # Create a unique cache key based on input
    cache_key = f"recommend:{ingredients}:{preferences}:{top_n}"

    # Check if the result is cached
    if redis_client:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logging.info(f"Cache HIT for key: {cache_key}")
            CACHE_HIT_COUNT.labels(endpoint="/recommend").inc()
            return ast.literal_eval(cached_result)  # Parse and return cached data
        CACHE_MISS_COUNT.labels(endpoint="/recommend").inc()

    # Process request (existing logic here)
    result = filter_recipes(ingredients, preferences, top_n)

    # Cache the result
    if redis_client:
        redis_client.set(cache_key, str(result), ex=3600)

    return result

@app.post("/recommend_by_embedding")
def recommend_by_embedding_endpoint(request: RecommendRequest):
    query = request.ingredients
    top_n = request.top_n

    # Create a unique cache key
    cache_key = f"recommend_by_embedding:{query}:{top_n}"

    # Check cache
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logging.info(f"Cache HIT for key: {cache_key}")
                return ast.literal_eval(cached_result)
            logging.info(f"Cache MISS for key: {cache_key}")
        except redis.RedisError as e:
            logging.error(f"Redis error during GET: {e}")

    # Compute recommendations
    query_vector = np.random.rand(embeddings.shape[1])  # Replace with actual embedding logic
    similarities = cosine_similarity([query_vector], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    result = df.iloc[top_indices][["title", "ingredients", "directions"]].to_dict(orient="records")

    for recipe in result:
        try:
            recipe["ingredients"] = ast.literal_eval(recipe["ingredients"])
            recipe["directions"] = ast.literal_eval(recipe["directions"])
        except Exception as e:
            logging.error(f"Error parsing recipe data: {e}")
            raise HTTPException(status_code=500, detail="Error processing recipe data.")

    # Cache the result
    if redis_client:
        try:
            redis_client.set(cache_key, str(result), ex=3600)
        except redis.RedisError as e:
            logging.error(f"Redis error during SET: {e}")

    return result

@app.post("/substitute")
def substitute_ingredients(request: SubstituteRequest):
    substitutions = suggest_substitutions(request.ingredient.lower())
    if not substitutions:
        raise HTTPException(status_code=404, detail="No substitutions found.")
    return {"substitutions": substitutions}
