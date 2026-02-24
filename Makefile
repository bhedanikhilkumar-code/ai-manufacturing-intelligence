PYTHON ?= python3

.PHONY: install test api dashboard synthetic

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	pytest -q

api:
	uvicorn aimi.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/aimi/dashboard.py --server.port 8501

synthetic:
	$(PYTHON) -m aimi.cli generate --batches 200 --output data/synthetic_batches.csv
