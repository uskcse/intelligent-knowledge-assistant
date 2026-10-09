#!/bin/sh
# Start the Ollama server; pull the configured model best-effort in the background.
# The server stays up even if the pull fails (no network / proxy), so the stack comes
# up regardless and the API degrades gracefully.
set -e

MODEL="${KA_OLLAMA_MODEL:-llama3.2:3b}"

# Trust a mounted corporate CA (if provided) so pulls work behind a TLS proxy.
if ls /usr/local/share/ca-certificates/*.crt >/dev/null 2>&1; then
  update-ca-certificates >/dev/null 2>&1 || true
fi

ollama serve &
server_pid=$!

echo "Waiting for Ollama server to accept connections..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

if ollama list | grep -q "$MODEL"; then
  echo "Model $MODEL already present."
else
  echo "Pulling model $MODEL in the background (best-effort)..."
  (
    attempt=0
    until ollama pull "$MODEL"; do
      attempt=$((attempt + 1))
      if [ "$attempt" -ge 3 ]; then
        echo "WARN: could not pull $MODEL after $attempt attempts; LLM answers unavailable until a model is pulled."
        break
      fi
      echo "pull attempt $attempt failed; retrying in 10s..."
      sleep 10
    done
  ) &
fi

wait "$server_pid"
