from unittest.mock import patch
from app.ingredient_logic import suggest_substitutions

def test_suggest_substitutions():
    substitutions = suggest_substitutions("sugar")
    assert substitutions == ["honey", "maple syrup"]

    substitutions = suggest_substitutions("milk")
    assert substitutions == ["almond milk", "soy milk"]

    substitutions = suggest_substitutions("unknown ingredient")
    assert substitutions == []
