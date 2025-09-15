# --- Stage 1: Builder ---
FROM python:3.12-slim as builder

# Set the working directory
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files needed for the build
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ ./src/

# Create a virtual environment and install all dependencies
RUN uv venv
RUN . .venv/bin/activate && uv sync --all-groups


# --- Stage 2: Final Image ---
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy only the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Set the PATH to use the Python and packages from the virtual environment
ENV PATH="/app/.venv/bin:${PATH}"

# Expose the port the app runs on
EXPOSE 8080

# Run the application with gunicorn
# We can call gunicorn directly because it's in the virtual environment's PATH
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "stock_market_analytics.inference.dashboard:server"]