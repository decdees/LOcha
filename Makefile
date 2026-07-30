.PHONY: dev test lint fmt check

dev:
	@echo "serving on http://127.0.0.1:8000"
	uv run uvicorn ocha.api.main:app --reload --host 127.0.0.1 --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test
