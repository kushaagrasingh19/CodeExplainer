from typing import List, Optional
from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    code: str = Field(..., description="The code block to analyze.")
    language: Optional[str] = Field(None, description="Optional language hint, e.g., 'python', 'javascript'.")
    question: Optional[str] = Field(
        default="Explain this code clearly and concisely.",
        description="Optional custom question to focus the explanation.",
    )
    top_k: int = Field(default=4, ge=0, le=10, description="How many context docs to retrieve from RAG.")


class RetrievedContext(BaseModel):
    source: Optional[str] = None
    score: Optional[float] = None
    snippet: str


class ExplainResponse(BaseModel):
    language: Optional[str]
    summary: str
    step_by_step: List[str]
    functions_and_components: List[str]
    complexity: Optional[str]
    potential_issues: List[str]
    improvement_suggestions: List[str]
    sample_usage: Optional[str]
    test_cases: List[str]
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)