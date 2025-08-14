import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Code Explainer", version="0.1.0")

# CORS (allow localhost and all by default; tighten as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from app.routes.explain import router as explain_router  # noqa: E402

app.include_router(explain_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}