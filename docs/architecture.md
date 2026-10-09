# Architecture

This document complements the top-level [README](../README.md) with detailed component, sequence,
and data-flow diagrams, plus module responsibilities.

## 1. Component overview

```mermaid
flowchart LR
    subgraph Client
        UI[Chainlit UI]
        CURL[REST clients]
    end

    subgraph Service["FastAPI service (stateless, horizontally scalable)"]
        API[Routes: /ask /health /ready /documents]
        MW[Middleware: request-id + latency + structured logs]
        SVC[QueryService]
        GRAPH[LangGraph agent]
        RETR[HybridRetriever]
        GEN[RagGenerator + grounding]
        PROV[Provider factory]
    end

    subgraph Backing["Backing services"]
        CHROMA[(Chroma vector DB)]
        OLLAMA[Ollama LLM]
    end

    subgraph Offline["Offline ingestion job"]
        ING[IngestionPipeline]
        LOAD[PyMuPDF loader]
        CHNK[Token chunker]
        MANI[Manifest]
    end

    UI --> API
    CURL --> API
    API --> MW --> SVC --> GRAPH
    GRAPH --> RETR
    GRAPH --> GEN
    RETR --> CHROMA
    GEN --> OLLAMA
    GRAPH --> OLLAMA
    PROV -.builds.-> GEN
    PROV -.builds.-> RETR

    ING --> LOAD --> CHNK --> CHROMA
    ING --> MANI
    CHNK -.embeddings.-> OLLAMA_EMB[sentence-transformers]
    RETR -.query embedding.-> OLLAMA_EMB
```

> Embeddings run **in-process** via sentence-transformers (not a separate service); the diagram
> separates them for clarity. The LLM (Ollama) is the only external model service.

## 2. Request sequence — agentic path

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as FastAPI /ask
    participant S as QueryService
    participant G as LangGraph
    participant R as HybridRetriever
    participant L as Ollama LLM

    C->>A: POST /ask {question, mode=auto}
    A->>S: answer(question, mode)
    S->>G: invoke(initial state)
    G->>L: router classify (SIMPLE/MULTI)
    L-->>G: MULTI
    Note over G: route = agentic
    G->>L: decompose into sub-questions
    L-->>G: [sub-q1, sub-q2]
    loop per sub-question
        G->>R: document_search(sub-q)
        R->>R: dense + BM25 → RRF → rerank
        R-->>G: ranked chunks
    end
    Note over G: aggregate + grounding gate
    G->>L: generate grounded answer (cited)
    L-->>G: answer with [n] citations
    G-->>S: final state (answer, citations, steps)
    S-->>A: QueryResult
    A-->>C: {answer, sources, workflow, steps, latency_ms}
```

For a **simple** question the router returns `rag`, and the graph runs a single
`retrieve → ground → generate` node before returning — no decomposition.

## 3. Ingestion data flow

```mermaid
flowchart LR
    A[PDF files in data/corpus] --> B{For each file}
    B --> C[file hash]
    C --> D{unchanged in manifest?}
    D -->|yes & not rebuild| SKIP[skip]
    D -->|no| E[PyMuPDF: text per page]
    E --> F[Token-aware chunk + metadata]
    F --> G[bge-small embeddings]
    G --> H[(Chroma upsert)]
    H --> I[update manifest]
```

Ingestion is **idempotent** (content-hash keyed) and **decoupled** from the query path. It runs as a
one-shot CLI (`ka-ingest run`) or a one-shot compose service.

## 4. Grounding & abstention decision

```mermaid
flowchart TD
    Q[Question] --> RT[Retrieve chunks]
    RT --> EMPTY{any chunks?}
    EMPTY -->|no| AB[Abstain: no_context]
    EMPTY -->|yes| TH{max score ≥ threshold?}
    TH -->|no| AB2[Abstain: below_score_threshold]
    TH -->|yes| OPT{LLM sufficiency check?}
    OPT -->|NO| AB3[Abstain: llm_insufficient]
    OPT -->|YES / skipped| GEN[Generate grounded answer]
    GEN --> SENT{answer is refusal sentinel?}
    SENT -->|yes| AB4[Abstain]
    SENT -->|no| OUT[Answer + citations]
```

## 5. Module responsibilities

| Module | Responsibility |
|---|---|
| `core/config.py` | Layered, typed settings (defaults < YAML < `.env` < env). Secrets only from env. |
| `core/logging.py` | Structured (structlog) logging; JSON in prod, console locally; request-id binding. |
| `core/models.py` | Internal domain models: `Chunk`, `ChunkMetadata`, `RetrievedChunk`, `Citation`. |
| `providers/` | `LLMProvider` / `EmbeddingProvider` protocols; Ollama, sentence-transformers, OpenAI impls; config-driven factory. |
| `ingestion/` | Loader, token-aware chunker, pipeline, idempotency manifest, CLI. |
| `vectorstore/` | Chroma HTTP wrapper: upsert/query/get_all/list/reset/health. |
| `retrieval/` | Dense, BM25, RRF fusion, cross-encoder rerank, `HybridRetriever`, metadata filters. |
| `rag/` | Prompts (injection-aware), grounded generator, grounding gate, citation assembly. |
| `agent/` | LangGraph state, router (heuristics + LLM), nodes, tools, graph builder. |
| `service/` | `QueryService` facade; builds and invokes the graph; maps to `QueryResult`. |
| `eval/` | Deterministic + LLM-judge metrics, runner, optional RAGAS, CLI. |
| `api/` | FastAPI app factory, routes, request/response schemas, error handling. |

## 6. Key design decisions (summary)

- **Local-first with a provider seam** — zero-key reproducibility now; trivial upgrade to hosted
  models later.
- **Hybrid retrieval + rerank** — semantics (dense) + exact terms (BM25) + precision (cross-encoder),
  fused with rank-based RRF to avoid score-scale coupling.
- **Unified graph for both paths** — routing, simple RAG, and the agentic flow live in one LangGraph
  state machine, giving consistent step/state tracking and a single place to reason about control
  flow.
- **Grounding as a first-class gate** — score threshold + injection-aware prompt + refusal sentinel,
  so "I don't know" is a designed outcome, not an accident.
- **Separation of concerns** — ingestion, retrieval, generation, agent, and transport are independent
  and independently testable (48 hermetic tests).
