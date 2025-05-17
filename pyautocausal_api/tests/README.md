# Tests for PyAutoCausal API

This directory contains tests for the PyAutoCausal API, including tests for the FastAPI endpoints and Celery worker tasks.

## Test Structure

- `test_main.py`: Tests for the FastAPI endpoints in `app/main.py`
- `test_worker.py`: Tests for the Celery worker tasks in `app/worker.py`
- `conftest.py`: Configuration for pytest

## Running the Tests

### Requirements

Make sure you have all development dependencies installed:

```bash
cd /path/to/pyautocausal_api
poetry install --with dev
```

### Running All Tests

To run all tests:

```bash
cd /path/to/pyautocausal_api
poetry run pytest tests/
```

### Running Specific Test Files

To run tests from a specific file:

```bash
poetry run pytest tests/test_main.py
poetry run pytest tests/test_worker.py
```

### Running Specific Tests

To run a specific test:

```bash
poetry run pytest tests/test_main.py::test_parse_s3_uri
```

### Running with Verbose Output

To run tests with more detailed output:

```bash
poetry run pytest -v tests/
```

### Code Coverage

To run tests with coverage reporting:

```bash
poetry run pytest --cov=app tests/
```

For a detailed HTML coverage report:

```bash
poetry run pytest --cov=app --cov-report=html tests/
```

This will create a `htmlcov` directory. Open `htmlcov/index.html` in a browser to view the report.

## Mocking Strategy

The tests extensively use pytest fixtures and the `unittest.mock` library to mock external dependencies like S3, Celery, and Redis. This allows the tests to run quickly and without requiring access to actual AWS services.

### Main Mocking Strategies:

1. **Environment Variables**: Mocked using fixtures to simulate different configurations
2. **S3 Client**: Mocked to simulate successful uploads/downloads as well as various error conditions
3. **Celery Task**: Mocked to simulate task submission and different task states
4. **File I/O**: Uses pytest's `tmpdir` fixture to create temporary directories and files

## Adding New Tests

When adding new tests:

1. Maintain the pattern of testing both the happy path and various error conditions
2. Use pytest fixtures for setup and teardown
3. Keep tests isolated from each other (don't let state leak between tests)
4. For API endpoints, use the FastAPI TestClient
5. For Celery tasks, mock the task self object and external dependencies 