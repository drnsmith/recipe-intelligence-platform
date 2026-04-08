def preprocess_ingredients(ingredients: str) -> str:
    """
    Preprocess ingredients by converting to lowercase
    and removing unnecessary punctuation or whitespace.
    """
    return " ".join(ingredients.lower().replace(",", " ").split())
