PYTHON ?= python
PKG := odontocal
SRC := src/$(PKG)

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "Targets:"
	@echo "  install  install the package (pip install -e .)"
	@echo "  dev      install with dev tools (pip install -e .[dev])"
	@echo "  lint     ruff + black --check + isort --check-only"
	@echo "  type     mypy --strict $(SRC)"
	@echo "  test     run the pytest suite"
	@echo "  smoke    odc-evaluate --help"
	@echo "  docker   build the Docker image"
	@echo "  clean    remove caches and build artifacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .

type:
	$(PYTHON) -m mypy --strict $(SRC)

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q tests

smoke:
	PYTHONPATH=src $(PYTHON) -m $(PKG).entry.evaluate --help

docker:
	docker build -t $(PKG):0.1 .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
