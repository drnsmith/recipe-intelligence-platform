"""
Llama3 recipe generation and constraint-aware adaptation via Ollama.
Handles dietary restrictions, skill level adjustment, and ingredient substitution.
"""

import requests
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3"
TIMEOUT = 120


class RecipeGenerator:
    """
    LLM-powered recipe generation and adaptation using Llama3 via Ollama.

    Capabilities:
    - Adapt an existing recipe to dietary constraints (vegan, gluten-free, etc.)
    - Adjust complexity for a target skill level (beginner / intermediate / advanced)
    - Substitute specific ingredients with alternatives
    - Generate a new recipe from a list of ingredients
    """

    def __init__(self, model: str = MODEL, ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is reachable."""
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            r.raise_for_status()
            logger.info(f"Ollama connected — using model: {self.model}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                f"Run `ollama serve` and ensure `{self.model}` is pulled. Error: {e}"
            )

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 1024
            }
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.Timeout:
            raise TimeoutError(f"Ollama did not respond within {TIMEOUT}s")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    def adapt_recipe(
        self,
        title: str,
        ingredients: str,
        directions: str,
        dietary: Optional[str] = None,
        skill_level: Optional[str] = None,
        substitutions: Optional[dict] = None
    ) -> dict:
        """
        Adapt an existing recipe to given constraints.

        Args:
            title: Recipe title
            ingredients: Original ingredients string
            directions: Original directions string
            dietary: Dietary constraint e.g. 'vegan', 'gluten-free', 'dairy-free'
            skill_level: Target skill level — 'beginner', 'intermediate', 'advanced'
            substitutions: Dict of {original_ingredient: substitute}

        Returns:
            Dict with adapted title, ingredients, directions, and notes
        """
        constraints = []
        if dietary:
            constraints.append(f"dietary requirement: {dietary}")
        if skill_level:
            constraints.append(f"target skill level: {skill_level}")
        if substitutions:
            subs = ", ".join(f"{k} → {v}" for k, v in substitutions.items())
            constraints.append(f"ingredient substitutions: {subs}")

        constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else "- none"

        prompt = f"""You are a professional chef and recipe developer.

Adapt the following recipe to meet the specified constraints. Keep the dish recognisable but ensure every constraint is fully satisfied. Return the result as JSON with keys: title, ingredients (list), directions (list of steps), notes (brief explanation of key changes made).

Original recipe:
Title: {title}
Ingredients: {ingredients}
Directions: {directions}

Constraints:
{constraint_text}

Return only valid JSON, no preamble."""

        logger.info(f"Adapting recipe: {title} | constraints: {constraints}")
        raw = self._call_llm(prompt)

        try:
            # Strip markdown code fences if present
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON — returning raw text")
            return {"title": title, "raw_response": raw, "parse_error": True}

    def generate_from_ingredients(
        self,
        ingredients: list[str],
        dietary: Optional[str] = None,
        skill_level: Optional[str] = None,
        cuisine: Optional[str] = None
    ) -> dict:
        """
        Generate a new recipe from a list of available ingredients.

        Args:
            ingredients: List of available ingredients
            dietary: Optional dietary constraint
            skill_level: Optional skill level
            cuisine: Optional cuisine style e.g. 'Italian', 'Asian'

        Returns:
            Dict with title, ingredients, directions, notes
        """
        ingredient_list = ", ".join(ingredients)
        constraints = []
        if dietary:
            constraints.append(f"dietary requirement: {dietary}")
        if skill_level:
            constraints.append(f"skill level: {skill_level}")
        if cuisine:
            constraints.append(f"cuisine style: {cuisine}")

        constraint_text = "\n".join(f"- {c}" for c in constraints) if constraints else "- none"

        prompt = f"""You are a professional chef and recipe developer.

Create a complete recipe using primarily the following ingredients. You may add basic pantry staples (salt, pepper, oil, water) but the listed ingredients should be the focus. Return the result as JSON with keys: title, ingredients (list), directions (list of steps), notes.

Available ingredients: {ingredient_list}

Constraints:
{constraint_text}

Return only valid JSON, no preamble."""

        logger.info(f"Generating recipe from {len(ingredients)} ingredients")
        raw = self._call_llm(prompt)

        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON — returning raw text")
            return {"raw_response": raw, "parse_error": True}


if __name__ == "__main__":
    gen = RecipeGenerator()

    # Test: adapt a recipe
    result = gen.adapt_recipe(
        title="Classic Beef Bolognese",
        ingredients="500g beef mince, 2 onions, 4 garlic cloves, 400g canned tomatoes, 200ml red wine, pasta",
        directions="Brown the mince. Soften onions and garlic. Add tomatoes and wine. Simmer 45 minutes. Serve with pasta.",
        dietary="vegan",
        skill_level="beginner",
        substitutions={"beef mince": "lentils", "red wine": "vegetable stock"}
    )
    print(json.dumps(result, indent=2))
