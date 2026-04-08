def suggest_substitutions(ingredient):
    """
    Suggest substitutions for a given ingredient.
    """
    substitution_map = {
        "butter": ["avocado", "coconut oil"],
        "milk": ["almond milk", "soy milk"],
        "sugar": ["honey", "maple syrup"],
    }
    return substitution_map.get(ingredient.lower(), [])
