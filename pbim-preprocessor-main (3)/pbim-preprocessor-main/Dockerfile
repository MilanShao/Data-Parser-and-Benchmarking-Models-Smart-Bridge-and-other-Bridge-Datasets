FROM python:3.10-slim as base
ENV PYTHONUNBUFFERED=1
ENV POETRY_CACHE_DIR=/tmp/.cache/pypoetry
ENV POETRY_VERSION=1.2.2

# System deps:
RUN apt update && apt install -y gcc
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-root

# Creating folders, and files for a project:
COPY . /app
RUN chmod o+w /app
