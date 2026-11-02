# Investment Research Assistant

An AI engineering project: a portfolio-aware research assistant. Cited answers from company reports (document + page), joined to the user's own portfolio transactions imported from broker CSV exports via LLM field mapping with a human review queue.

**Stack:** Python, Docker Compose, LlamaIndex, LangChain/LangGraph, Elasticsearch, Postgres + pgvector, Streamlit.

## Setup

```bash
cp .env.example .env   # add your API keys
docker compose up -d
```

Uses sample or synthetic data only.
