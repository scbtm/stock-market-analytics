FROM python:3.12-slim

WORKDIR /app

# Install build tools and uv
RUN apt-get update && apt-get install -y curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy all project files
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ ./src/

# Install all dependencies.
# This creates a .venv and installs the project in it.
RUN uv sync --all-groups

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose the port the app runs on
EXPOSE 8080

# Default entrypoint - can be overridden via ENTRYPOINT_COMMAND env var
ENV ENTRYPOINT_COMMAND=dashboard

# Use entrypoint script to handle different commands
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
