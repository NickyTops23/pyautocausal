# Simplified single-stage Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.5 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Set the working directory
WORKDIR /app

# Copy the pyautocausal package
COPY pyautocausal /pyautocausal

# Copy the API project files
COPY pyautocausal_api/pyproject.toml pyautocausal_api/poetry.lock ./

# Debug: List contents to verify
RUN ls -la && echo "===== PYPROJECT.TOML CONTENTS =====" && cat pyproject.toml && \
    echo "===== PYAUTOCAUSAL DIRECTORY =====" && ls -la /pyautocausal

# Install dependencies without running poetry lock (using existing lock file)
RUN poetry install --no-dev --no-root

# Copy the application code
COPY pyautocausal_api/app ./app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]