import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import ast

# Load dataset
# Load dataset
def load_data():
    file_path = "full_dataset.csv"
    df = pd.read_csv(file_path, nrows=100000)
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['NER'] = df['NER'].apply(ast.literal_eval)

    # Convert list columns to strings for Dash compatibility
    df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['NER'] = df['NER'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    return df

df = load_data()


# Extract categories (Assuming categories are inferred from titles)
df['category'] = df['title'].apply(lambda x: 'Dessert' if 'cake' in x.lower() or 'cookie' in x.lower() else 'Main Course')

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🍽 AI-Powered Recipe Dashboard", style={'textAlign': 'center'}),

    # Recipe Search by Title
    dcc.Input(id="title_search", type="text", placeholder="🔍 Search by Title...", debounce=True),
    html.Br(),
    dash_table.DataTable(
        id='recipe_table',
        columns=[{"name": i, "id": i} for i in ['title', 'ingredients']],
        page_size=10,
        style_table={'overflowX': 'auto'}
    ),

    # Recipe Category Dropdown
    html.Label("📂 Filter by Category:"),
    dcc.Dropdown(
        id='category_filter',
        options=[{"label": cat, "value": cat} for cat in df['category'].unique()],
        multi=True
    ),

    # Ingredient Frequency Visualization
    html.H2("📊 Most Common Ingredients"),
    dcc.Graph(id='ingredient_graph'),

    # Search Recipes by Ingredient
    dcc.Input(id="ingredient_search", type="text", placeholder="🥕 Search by Ingredient...", debounce=True),
    html.Br(),
    dash_table.DataTable(
        id='ingredient_table',
        columns=[{"name": i, "id": i} for i in ['title', 'ingredients']],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )
])

# Callbacks for interactivity
@app.callback(
    Output('recipe_table', 'data'),
    Input('title_search', 'value')
)
def update_table(search_query):
    if search_query:
        filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
        return filtered_df[['title', 'ingredients']].to_dict('records')
    return df[['title', 'ingredients']].head(10).to_dict('records')

@app.callback(
    Output('ingredient_graph', 'figure'),
    Input('category_filter', 'value')
)
def update_ingredient_chart(selected_categories):
    filtered_df = df if not selected_categories else df[df['category'].isin(selected_categories)]
    all_ingredients = [ingredient for sublist in filtered_df['NER'].dropna() for ingredient in sublist]
    ingredient_counts = pd.Series(all_ingredients).value_counts().head(15)
    
    fig = px.bar(ingredient_counts, x=ingredient_counts.index, y=ingredient_counts.values, title="Most Common Ingredients", labels={'x': 'Ingredient', 'y': 'Count'})
    return fig

@app.callback(
    Output('ingredient_table', 'data'),
    Input('ingredient_search', 'value')
)
def update_ingredient_table(search_query):
    if search_query:
        filtered_df = df[df['NER'].apply(lambda x: search_query.lower() in [i.lower() for i in x])]
        return filtered_df[['title', 'ingredients']].to_dict('records')
    return df[['title', 'ingredients']].head(10).to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)
