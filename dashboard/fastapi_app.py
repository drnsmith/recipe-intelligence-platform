from fastapi import FastAPI, Query
import pandas as pd
import ast

# Load dataset
def load_data():
    file_path = "full_dataset.csv"
    df = pd.read_csv(file_path, nrows=100000)
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['NER'] = df['NER'].apply(ast.literal_eval)

    # Convert list columns to strings for API responses
    df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['NER'] = df['NER'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    return df

df = load_data()

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Recipe API!"}

# Get all recipes (Paginated)
@app.get("/recipes/")
def get_recipes(skip: int = 0, limit: int = 10):
    return df.iloc[skip: skip + limit].to_dict(orient="records")

# Search recipes by title
@app.get("/recipes/search/")
def search_recipes(title: str = Query(..., description="Search by recipe title")):
    results = df[df['title'].str.contains(title, case=False, na=False)]
    return results.to_dict(orient="records")

# Search recipes by ingredient
@app.get("/recipes/by-ingredient/")
def search_by_ingredient(ingredient: str = Query(..., description="Search recipes containing this ingredient")):
    results = df[df['NER'].apply(lambda x: ingredient.lower() in x.lower())]
    return results.to_dict(orient="records")

