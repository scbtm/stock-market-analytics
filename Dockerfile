# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Copy the application code
COPY src/ ./src/

# Install dependencies
RUN uv sync --all-groups

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["uv", "run", "gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "stock_market_analytics.inference.dashboard:server"]
