import openai

# Set up OpenAI API Key (or replace with open-source LLM later)
OPENAI_API_KEY = "your-api-key-here"

def generate_sql_query(user_input):
    """
    Uses an LLM to convert natural language into an SQL query.
    """
    prompt = f"Convert this request into an SQL query for a recipe database: {user_input}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use an open-source model
        messages=[{"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )

    sql_query = response["choices"][0]["message"]["content"]
    return sql_query
