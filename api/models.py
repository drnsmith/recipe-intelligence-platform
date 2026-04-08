from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    ingredients: Optional[str] = None
    preferences: Optional[List[str]] = None
    top_n: int = 3
