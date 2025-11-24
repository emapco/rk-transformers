.PHONY: uv-export
uv-export:
	uv pip install -e .[export]

.PHONY: uv-inference
uv-inference:
	uv pip install -e .[inference]

.PHONY: dev
uv-dev:
	uv pip install -e .[dev]

.PHONY: test
test:
	pytest tests -v

.PHONY: test-cov
test-cov:
	pytest --cov=rktransformers --cov-report=html --cov-report=term-missing tests -v

.PHONY: lint
lint:
	ruff check . --fix
	ruff format . tests
	ruff analyze graph
	ruff clean
