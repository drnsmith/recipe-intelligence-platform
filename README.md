# Recipe Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-caching-DC382D?style=flat-square&logo=redis&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-deployed-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end NLP and LLM pipeline for large-scale recipe intelligence, built on the RecipeNLG corpus (2.2M recipes). The platform combines difficulty classification, semantic retrieval, and LLM-powered recipe generation into a unified production system — deployed via FastAPI with Redis caching, Prometheus monitoring, and a Streamlit dashboard.

**Live demo:** [huggingface.co/spaces/drnsmith/recipe-intelligence](https://huggingface.co/spaces/drnsmith/recipe-intelligence)

---

## Overview

Most recipe applications treat food as a search problem. This platform treats it as an intelligence problem: understanding what makes a recipe complex, surfacing semantically similar recipes from a corpus of 2.2M, adapting them through an LLM based on constraints (dietary, skill level, available ingredients), and doing all of this through a production-grade API with monitoring and drift detection.

The system is designed around a real business question: how do you build a reliable, explainable, production-ready AI system on messy, large-scale culinary text data?

---

## Architecture

```
RecipeNLG corpus (2.2M recipes)
         │
         ▼
┌─────────────────────┐
│  Data pipeline       │  Cleaning · complexity scoring · clustering
│                      │  recipes_with_complexity.csv · clustered_recipe.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Classifier layer    │  10-model NLP benchmark (see AI-Recipe-Classifier)
│                      │  Custom NN · F1: 0.753 · LIME explainability
│                      │  Easy · Medium · Hard · Very Hard
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Embedding layer     │  sentence-transformers/all-MiniLM-L6-v2
│                      │  FAISS index over full 2.2M corpus
│                      │  Semantic retrieval · MMR diversity reranking
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Generation layer    │  Llama3 via Ollama
│                      │  Constraint-aware recipe adaptation
│                      │  Dietary · skill level · ingredient substitution
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Serving layer       │  FastAPI · Redis caching · Prometheus metrics
│                      │  Evidently drift monitoring
│                      │  Streamlit dashboard for exploration
└─────────────────────┘
```

---

## Key Features

- **Large-scale NLP** — classification and retrieval pipeline operating on 2.2M recipes, one of the largest publicly available culinary corpora
- **Difficulty classification** — custom neural network (F1: 0.753) with LIME explainability, trained on domain-engineered complexity features
- **Semantic retrieval** — sentence-transformer embeddings with FAISS for sub-second nearest-neighbour search across the full corpus
- **LLM adaptation** — Llama3-powered recipe generation and constraint-aware adaptation (dietary restrictions, skill level, ingredient substitution)
- **Production monitoring** — Redis caching, Prometheus metrics, and Evidently data drift detection for reliability in deployment
- **Interactive dashboard** — Streamlit interface for recipe exploration, difficulty filtering, and similarity search

---

## Tech Stack

| Component | Technology |
|---|---|
| Corpus | RecipeNLG (2.2M recipes, Poznań University of Technology) |
| Classification | scikit-learn, TensorFlow, XGBoost |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | FAISS |
| LLM | Llama3 via Ollama |
| Backend | FastAPI, Python 3.11 |
| Caching | Redis |
| Monitoring | Prometheus, Evidently |
| Dashboard | Streamlit |
| Deployment | Docker, Hugging Face Spaces |

---

## Project Structure

```
recipe-intelligence-platform/
├── data/
│   └── sample_recipes.csv         # 1k-row sample for reproducibility
├── classifier/
│   ├── complexity_features.py     # Domain-engineered label assignment
│   ├── train.py                   # 10-model benchmark training
│   └── explain.py                 # LIME explainability
├── retrieval/
│   ├── embeddings.py              # Sentence-transformer encoding
│   ├── index.py                   # FAISS index builder
│   └── search.py                  # Semantic search + MMR reranking
├── generation/
│   └── llm_adapter.py             # Llama3 recipe adaptation via Ollama
├── api/
│   ├── main.py                    # FastAPI application
│   ├── routes.py                  # Endpoints
│   └── cache.py                   # Redis integration
├── monitoring/
│   ├── drift.py                   # Evidently drift detection
│   └── prometheus.yml             # Metrics configuration
├── dashboard/
│   └── app.py                     # Streamlit interface
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/drnsmith/recipe-intelligence-platform.git
cd recipe-intelligence-platform
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build FAISS index (uses sample data by default)
python retrieval/index.py

# Start API
uvicorn api.main:app --reload --port 8000

# Start dashboard (separate terminal)
streamlit run dashboard/app.py
```

API docs: `http://127.0.0.1:8000/docs`
Dashboard: `http://127.0.0.1:8501`

### Docker

```bash
docker-compose up --build
```

---

## API Reference

### `POST /classify`
Predict difficulty level for a recipe.

```json
{"title": "Beef Wellington", "ingredients": ["beef tenderloin", "puff pastry", "mushrooms"], "instructions": ["Sear the beef...", "Wrap in mushroom duxelles..."]}
```

**Response:** `{"difficulty": "Hard", "confidence": 0.84, "explanation": {"top_features": ["wrap", "duxelles", "sear"]}}`

### `POST /search`
Semantic recipe search.

```json
{"query": "quick vegetarian pasta under 30 minutes", "k": 5, "difficulty_filter": "Easy"}
```

### `POST /generate`
LLM-powered recipe adaptation.

```json
{"base_recipe_id": 12345, "constraints": {"dietary": "vegan", "skill_level": "beginner", "substitute": {"butter": "coconut oil"}}}`
```

---

## Dataset

**RecipeNLG** — Bień et al. (2020). *RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation.* INLG 2020. Poznań University of Technology. Available at [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/)

---

## Roadmap — v2: Multimodal Extension

The next phase of this platform adds cross-modal image-recipe intelligence using the Recipe1M dataset (800k+ food images, MIT CSAIL):

- **Image → recipe retrieval** — CLIP embeddings for food image understanding, FAISS retrieval over the text corpus
- **EfficientNet food classifier** — fine-tuned on Recipe1M images for category and difficulty prediction from photos
- **Cross-modal search** — upload a food photo, retrieve the most similar recipes from 2.2M

Data is staged and ready. Implementation in progress in the [Multimodal-Food-AI](https://github.com/drnsmith/multimodal-food-ai) repository.

---

## Related Projects

- [AI-Recipe-Classifier](https://github.com/drnsmith/AI-Recipe-Classifier) — standalone NLP benchmark for the difficulty classification component
- [Multimodal-Food-AI](https://github.com/drnsmith/multimodal-food-ai) — cross-modal image-recipe retrieval (v2)

---

## Credits

Built by [@drnsmith](https://github.com/drnsmith) — quantitative data scientist specialising in end-to-end ML systems.

[Medium](https://medium.com/@NeverOblivious) · [Substack](https://substack.com/@errolog) · [LinkedIn](https://linkedin.com/in/drnsmith)
