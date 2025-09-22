#!/bin/bash

# Default to dashboard if no entrypoint command specified
ENTRYPOINT_COMMAND=${ENTRYPOINT_COMMAND:-dashboard}

case "$ENTRYPOINT_COMMAND" in
    "dashboard")
        exec uv run gunicorn -w 4 -b 0.0.0.0:8080 stock_market_analytics.inference.dashboard:server
        ;;
    "batch-collect")
        exec uv run batch-collect run
        ;;
    "build-features")
        exec uv run build-features run
        ;;
    "train-model")
        exec uv run train-model run
        ;;
    "monitor-model")
        exec uv run monitor-model run
        ;;
    *)
        echo "Unknown entrypoint command: $ENTRYPOINT_COMMAND"
        echo "Defaulting to dashboard..."
        exec uv run gunicorn -w 4 -b 0.0.0.0:8080 stock_market_analytics.inference.dashboard:server
        ;;
esac