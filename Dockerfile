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

# Expose the port the app runs on
EXPOSE 8080

# Run the application using `uv run` to ensure it uses the virtual environment
CMD ["uv", "run", "gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "stock_market_analytics.inference.dashboard:server"]
