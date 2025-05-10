# Dockerfile

# Stage 1: Build the Poetry environment and install dependencies
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.0 \
    # Tell poetry to not create a virtual environment, as the container is the environment
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Set the working directory
WORKDIR /app

# Copy only the pyproject.toml file and generate the lock file
# This leverages Docker's cache. If these files don't change, this layer won't be rebuilt.
COPY pyproject.toml ./
RUN poetry lock

# Install project dependencies (without dev dependencies)
# --no-root is used because we will copy the application code in the next stage
# or in the final stage, ensuring poetry installs the project itself if it's structured as a package.
# If your project is not installed as an editable package locally, you might not need --no-root
# and can just run `poetry install --no-dev`
RUN poetry install --no-dev --no-root

# Stage 2: Create the final, lean production image
FROM python:3.10-slim as final

# Set environment variables (can be repeated or inherited if not changed)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the virtual environment with dependencies from the builder stage
COPY --from=builder /app /app
# ... (rest of comments and explanation from previous Dockerfile version regarding dependency copying) ...

# Copy the application code
# If your code is in an 'app' subdirectory:
COPY ./app ./app
# If your code is directly in the root (e.g. main.py is in the root alongside pyproject.toml):
# COPY . .
# If your main package is pyautocausal_api
# COPY ./pyautocausal_api ./pyautocausal_api
# COPY main.py . # or wherever your entrypoint script is

# Ensure you copy any other necessary files or directories like templates, static files, etc.
# e.g., COPY ./static ./static

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
# Replace `app.main:app` with the correct path to your FastAPI app instance
# e.g., if your FastAPI instance `app` is in `my_project/api/server.py`, it would be `my_project.api.server:app`
# Assuming your FastAPI app instance is named `app` in `app/main.py`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]