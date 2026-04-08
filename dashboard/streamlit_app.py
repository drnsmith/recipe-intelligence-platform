import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt

# Load dataset with caching
@st.cache_data
def load_data():
    file_path = "full_dataset.csv"  # Update this if needed
    df = pd.read_csv(file_path, nrows=100000)  # Load a sample for speed
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['NER'] = df['NER'].apply(ast.literal_eval)
    return df

df = load_data()

# Streamlit UI
st.title("🍽 AI-Powered Recipe Explorer")

# Search by title
search_query = st.text_input("🔍 Search Recipe Title", "")
if search_query:
    filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
    st.write(filtered_df[['title', 'ingredients', 'directions']])

# Ingredient Frequency Visualization
st.subheader("📊 Most Common Ingredients")
all_ingredients = [ingredient for sublist in df['NER'].dropna() for ingredient in sublist]
ingredient_counts = pd.Series(all_ingredients).value_counts().head(15)

fig, ax = plt.subplots()
ingredient_counts.plot(kind="bar", ax=ax, color="skyblue")
st.pyplot(fig)

# Search Recipes by Ingredient
ingredient_search = st.text_input("🥕 Search by Ingredient", "")
if ingredient_search:
    matching_recipes = df[df['NER'].apply(lambda x: ingredient_search.lower() in [i.lower() for i in x])]
    st.write(matching_recipes[['title', 'ingredients', 'directions']])
