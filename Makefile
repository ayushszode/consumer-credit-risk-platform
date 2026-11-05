.PHONY: setup setup-extras demo run run-spark sample test api app clean

setup:            ## install everything into the current environment
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

setup-extras:     ## optional: XGBoost, PyTorch (CPU build) and PySpark
	pip install -r requirements-extras.txt --extra-index-url https://download.pytorch.org/whl/cpu

demo:             ## full pipeline on synthetic data (no download needed)
	python -m credit_risk.pipeline --synthetic

sample:           ## 20% stratified sample of the real data, for quick checks
	python -m credit_risk.pipeline --sample 0.2

run:              ## full pipeline on the real LendingClub file
	python -m credit_risk.pipeline

run-spark:        ## full pipeline, with the data preparation done in PySpark (needs Java 17+)
	python -m credit_risk.pipeline --engine spark

test:
	pytest

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf artifacts/*.joblib reports/results/* reports/figures/* data/processed/*
