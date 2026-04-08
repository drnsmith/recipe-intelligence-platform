from app.utils import preprocess_ingredients

def test_preprocess_ingredients():
    input_text = "Tomatoes, Garlic, Olive Oil"
    expected_output = "tomatoes garlic olive oil"
    assert preprocess_ingredients(input_text) == expected_output
