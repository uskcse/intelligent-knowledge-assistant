#!/usr/bin/env bash
# Portable launcher: auto-detects hardware and starts the stack optimally.
#
# - NVIDIA GPU (Linux/WSL2)  -> enables GPU acceleration for Ollama
# - Apple Silicon            -> CPU container (Metal isn't available to containers);
#                               prints the optional native-Ollama path for Metal
# - anything else            -> CPU-only (the always-portable default)
#
# The plain `docker compose up --build` remains a valid, always-works fallback.
set -euo pipefail

cd "$(dirname "$0")/.."

FILES=(-f docker-compose.yml)
OS="$(uname -s)"
ARCH="$(uname -m)"

if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
  echo ">> NVIDIA GPU detected — enabling GPU acceleration for Ollama."
  FILES+=(-f docker-compose.gpu.yml)
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  echo ">> Apple Silicon detected — containers cannot use Metal, running CPU."
  echo "   For Metal speedups, run Ollama natively and set:"
  echo "     KA_OLLAMA_BASE_URL=http://host.docker.internal:11434"
else
  echo ">> No supported GPU detected — running CPU-only (portable default)."
fi

# Corporate proxy support (optional, no-op when unset).
if [ -n "${HTTPS_PROXY:-}${HTTP_PROXY:-}" ]; then
  echo ">> Proxy env detected — passing HTTP(S)_PROXY through to Ollama pulls."
fi
if [ -f certs/corp-ca.crt ]; then
  echo ">> certs/corp-ca.crt found — trusting the corporate CA in the Ollama container."
  FILES+=(-f docker-compose.proxy.yml)
fi

exec docker compose "${FILES[@]}" up --build "$@"
