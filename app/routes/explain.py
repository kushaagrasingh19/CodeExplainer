from fastapi import APIRouter, HTTPException

from app.models.schemas import ExplainRequest, ExplainResponse
from app.services.explainer import ExplainerService

router = APIRouter()


def get_service() -> ExplainerService:
    # Lazy init per process
    # Note: This is simple; for production, consider dependency injection or lifespan events
    if not hasattr(get_service, "_svc"):
        get_service._svc = ExplainerService()  # type: ignore[attr-defined]
    return get_service._svc  # type: ignore[attr-defined]


@router.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    try:
        svc = get_service()
        return svc.explain_code(
            code=req.code,
            language=req.language,
            question=req.question,
            top_k=req.top_k,
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))