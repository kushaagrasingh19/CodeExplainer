import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from app.services.explainer import ExplainerService
from app.rag.indexer import ingest_path


load_dotenv()

st.set_page_config(page_title="AI Code Explainer", layout="wide")


@st.cache_resource
def get_service() -> ExplainerService:
	return ExplainerService()


def parse_extensions(csv_value: str) -> Optional[List[str]]:
	csv_value = (csv_value or "").strip()
	if not csv_value:
		return None
	return [p.strip().lower() for p in csv_value.split(",") if p.strip()]


st.title("AI Code Explainer")
st.caption("LangChain + OpenAI + ChromaDB (RAG)")

if not os.getenv("OPENAI_API_KEY"):
	st.warning("OPENAI_API_KEY is not set. Add it to .env or your environment.")

with st.sidebar:
	st.header("Settings")
	language_hint = st.text_input("Language hint (optional)", value="")
	question = st.text_area(
		"Question",
		value="Explain this code clearly and concisely. Highlight pitfalls and improvements.",
		height=80,
	)
	top_k = st.slider("Top-K context documents", min_value=0, max_value=10, value=4)

	st.divider()
	st.subheader("Optional: Ingest directory into Chroma")
	ingest_dir = st.text_input("Directory path")
	ext_csv = st.text_input(
		"Extensions CSV",
		value="py,js,ts,tsx,java,go,rs,cpp,c,h,md",
		help="Empty = default set",
	)
	if st.button("Ingest directory"):
		with st.spinner("Indexing... this may take a while for large repos"):
			count = ingest_path(ingest_dir, extensions=parse_extensions(ext_csv)) if ingest_dir else 0
		st.success(f"Indexed {count} chunks into ChromaDB.")

code = st.text_area(
	"Paste code here",
	height=400,
	placeholder="""// Example (Python)\n\n"""
)

col1, col2 = st.columns([1, 1])

with col1:
	if st.button("Explain", type="primary"):
		if not code.strip():
			st.error("Please paste some code first.")
			st.stop()
		with st.spinner("Thinking..."):
			service = get_service()
			resp = service.explain_code(
				code=code,
				language=language_hint or None,
				question=question or None,
				top_k=top_k,
			)
			st.session_state["last_response"] = resp

with col2:
	resp = st.session_state.get("last_response")
	if resp:
		st.subheader("Summary")
		st.write(resp.summary)

		st.subheader("Language")
		st.write(resp.language or language_hint or "Unknown")

		st.subheader("Step by step")
		for step in resp.step_by_step:
			st.markdown(f"- {step}")

		st.subheader("Functions & Components")
		for item in resp.functions_and_components:
			st.markdown(f"- {item}")

		st.subheader("Potential issues")
		for item in resp.potential_issues:
			st.markdown(f"- {item}")

		st.subheader("Improvement suggestions")
		for item in resp.improvement_suggestions:
			st.markdown(f"- {item}")

		if resp.sample_usage:
			st.subheader("Sample usage")
			st.code(resp.sample_usage, language=resp.language or "text")

		st.subheader("Test cases")
		for item in resp.test_cases:
			st.markdown(f"- {item}")

		st.subheader("Retrieved context")
		if not resp.retrieved_context:
			st.caption("No context retrieved. Try ingesting your repo for better results.")
		else:
			for c in resp.retrieved_context:
				source = c.source or "unknown"
				score = f"{c.score:.4f}" if c.score is not None else "n/a"
				with st.expander(f"{source} (score: {score})"):
					st.code(c.snippet, language="text")