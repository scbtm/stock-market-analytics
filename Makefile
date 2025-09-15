.PHONY: verify format lint typecheck test view-test-report

PY := python

format:
	uv run ruff format src tests

lint:
	uv run ruff check --fix src

typecheck:
	uv run pyright

test:
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing

view-test-report:
	uv run coverage html && open htmlcov/index.html

security-check:
	uv sync --all-groups
	uv run --with pip-audit pip-audit --local --skip-editable -v

verify: format typecheck test security-check lint

pre-commit: format typecheck lint

ci-tests:
	uv run pytest -m "not ci-exclude"

ci-verify: format lint typecheck ci-tests security-check