PYTHON := ./venv/bin/python

.PHONY: test test-v test-tns lint

test:
	$(PYTHON) -m pytest

test-v:
	$(PYTHON) -m pytest -v

test-tns:
	$(PYTHON) -m pytest tests/test_tns_sage.py -v

lint:
	$(PYTHON) -m ruff check .
