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

.PHONY: docs
docs: docs-clean
	cd docs && $(MAKE) html

.PHONY: docs-clean
docs-clean:
	cd docs && $(MAKE) clean

.PHONY: docs-serve
docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	python3 -m http.server --directory docs/build/html 8000
