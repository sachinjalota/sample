# Variables
PYTHON = python
POETRY = poetry
PROJECT_NAME = genai-guardrails-service
SOURCE_DIR = src
TESTS_DIR = tests
DIST_DIR = dist

# Lint and format
lint:
    @echo "Running ruff for lint..."
    $(POETRY) run ruff check $(SOURCE_DIR) $(TESTS_DIR)

format:
    @echo "Running isort and ruff format for code formatting..."
    $(POETRY) run isort $(SOURCE_DIR) $(TESTS_DIR)
    $(POETRY) run ruff format $(SOURCE_DIR) $(TESTS_DIR)

# TESTS
test:
    @echo "Running tests with pytest..."

# BUILD
build:
    @echo "Building the project with poetry..."
    $(POETRY) build

# Clean
clean:
    @echo "Cleaning up build and distribution directories..."
    rm -rf $(DIST_DIR) *.egg-info

# Setup virtual env
setup:
    @echo "Setting up the virtual env for project..."
    $(POETRY) config virtualenv.in-project true
    poetry install
    pre-commit install

# Run the application
run:
    @echo "Running the application..."

lint-format-
