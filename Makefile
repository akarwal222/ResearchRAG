.RECIPEPREFIX = >
PY = python

.PHONY: install download serve ui ingest eval test

install:
> pip install -r requirements.txt && pip install -e .

download:
> $(PY) scripts/download_papers.py

serve:
> $(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
> $(PY) -m ui.gradio_app

ingest:
> curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{}'

eval:
> $(PY) -m evaluation.retrieval_eval

test:
> $(PY) -m pytest -q
