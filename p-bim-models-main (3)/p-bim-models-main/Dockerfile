FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.2

ENV POETRY_CACHE_DIR=/tmp/.cache/pypoetry
# Wandb setup
RUN touch /.netrc && chmod 006 /.netrc

# System deps:
RUN apt update && apt install -y gcc vim
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-root

# Creating folders, and files for a project:
COPY . /app
RUN chmod -R o+w /app