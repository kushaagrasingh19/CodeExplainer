import json
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models.schemas import ExplainResponse
from app.rag.vectorstore import get_vectorstore


SYSTEM_PROMPT = (
    "You are a senior software engineer and educator. "
    "Given a code block and optional context from a codebase, produce a concise, accurate explanation. "
    "Explain in clear steps, identify key functions/components, complexity, pitfalls, and improvement ideas. "
    "If the language is unknown, infer it from the code. Answer only based on the code and provided context."
)


def format_context(context_docs: List[Dict[str, Any]]) -> str:
    if not context_docs:
        return ""
    blocks: List[str] = []
    for idx, item in enumerate(context_docs, start=1):
        source = item.get("source") or "unknown"
        snippet = item.get("snippet") or ""
        blocks.append(f"[Context {idx}] Source: {source}\n{snippet}")
    return "\n\n".join(blocks)


class ExplainerService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = ChatOpenAI(model=self.settings.openai_model, temperature=self.settings.temperature)
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\nAlways respond with a single JSON object with the following keys: "
                        "language, summary, step_by_step, functions_and_components, complexity, "
                        "potential_issues, improvement_suggestions, sample_usage, test_cases, retrieved_context."),
            ("human", "Language hint: {language}\nQuestion: {question}\n\nCode:\n```\n{code}\n```\n\nContext (optional):\n```\n{context_block}\n```\n\nReturn strictly valid JSON. Do not include markdown fences.")
        ])

    def retrieve_context(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query_text, k=top_k)
        formatted: List[Dict[str, Any]] = []
        for doc, score in results:
            formatted.append({
                "source": doc.metadata.get("source"),
                "score": float(score),
                "snippet": doc.page_content[:1800],
            })
        return formatted

    def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        question: Optional[str] = None,
        top_k: int = 4,
    ) -> ExplainResponse:
        question_text = question or "Explain this code clearly and concisely."
        context = self.retrieve_context(code, top_k=top_k)
        context_block = format_context(context)

        chain = self.prompt | self.llm | self.parser
        raw = chain.invoke({
            "language": language or "",
            "question": question_text,
            "code": code,
            "context_block": context_block,
        })

        # Ensure we have a dict (JsonOutputParser already returns one)
        payload: Dict[str, Any] = raw if isinstance(raw, dict) else json.loads(str(raw))

        # Attach retrieved context passthrough
        payload["retrieved_context"] = context

        return ExplainResponse.model_validate(payload)