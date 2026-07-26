# NexTripAI-BE

Backend service for NexTripAI. This repo owns the API layer, AI agent orchestration, user/session data, and integration with the Knowledge Base, Maps, Weather, and Frontend.

## Workflow Documentation

Read this before implementation:

- Repo workflow: [docs/WORKFLOW.md](docs/WORKFLOW.md)
- Logging: [docs/LOGGING.md](docs/LOGGING.md)

This backend should be local-first: FastAPI for APIs, LangGraph for the agent workflow, and HTTP clients for KB/weather integrations. Do not copy Azure/Foundry/Teams infrastructure from the reference Orchestrator project.

## Main Responsibilities

- Expose REST/WebSocket APIs for the frontend chat and trip planning flows.
- Orchestrate multi-agent workflows with LangGraph or a similar agent framework.
- Parse user intent, extract travel constraints, and manage conversation state.
- Call the Knowledge Base service/repo for GraphRAG retrieval.
- Run recommendation and itinerary planning logic.
- Integrate external APIs such as weather and maps/routing.
- Store user profiles, chat sessions, preferences, and feedback.

## Suggested Tech Stack

- Python
- FastAPI
- LangChain / LangGraph
- PostgreSQL
- Redis for cache/session state, optional
- Docker

## Suggested Structure

```txt
NexTripAI-BE/
  app/
    api/
    agents/
    core/
    models/
    services/
    schemas/
  tests/
  .env.example
  requirements.txt
  README.md
```

## System Boundary

The frontend should only communicate with this backend. The backend is responsible for calling the Knowledge Base and external services.

```txt
NexTripAI-FE -> NexTripAI-BE -> NexTripAI-KB
                         |
                         +-> Weather API
                         +-> Maps API
```

## Initial API Ideas

- `POST /chat`: send a user message and receive an agent response.
- `POST /trips/plan`: generate an itinerary from user constraints.
- `GET /sessions/{session_id}`: get conversation/session state.
- `POST /feedback`: collect user feedback for evaluation.

## Development Notes

- Keep prompts, agent graph definitions, and tool wrappers versioned.
- Log retrieved evidence from the KB to reduce hallucination risk.
- Keep response schemas stable so the frontend can render itinerary, recommendations, and citations consistently.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn src.app:app --reload --port 8000
```

Implemented endpoints:

- `GET /live` and `GET /ready`
- `GET /health`
- `POST /api/chat`
- `POST /api/chat/stream` (SSE events: `accepted`, `heartbeat`, `result`, `error`)
- `GET /api/sessions/{session_id}/messages`
- `DELETE /api/sessions/{session_id}`

The chat pipeline calls the KB over HTTP, optionally checks Open-Meteo, and stores
conversation messages plus rolling memory in memory or Firestore. It never connects
directly to Neo4j. Local Firestore authentication can use
`FIRESTORE_CREDENTIALS_PATH`; deployed workloads should use an attached service account.

Current chat orchestration:

```text
Frontend -> POST /api/chat -> Conversation Context Resolver
                            -> Gemini Contextualizer
                               +-> conversation recall -> transcript answer
                               +-> travel follow-up -> standalone request
                            -> TravelOrchestrator
                               +-> GraphRAG Agent -> KB API -> Neo4j evidence
                               +-> Weather Agent -> Open-Meteo assessment
                            -> Answer Synthesizer -> grounded Vietnamese response
```

The orchestrator selects `graph_only`, `weather_only`, or `graph_and_weather`. GraphRAG and
Weather run concurrently when both are required. The answer synthesizer receives the combined
tool context once; if a dependency is unavailable, a deterministic response reports the missing
tool without discarding successful evidence from the other branch.

Agent production capabilities:

- Durable transcript, structured trip state, and rolling summary backed by the chat store.
- Generic reference resolution and conversation recall without phrase-specific hard coding.
- Fail-open contextualization: Gemini failures keep the original travel request usable.
- Explicit tool routing, concurrent execution, timeout/error isolation, and clarification states.
- Grounded synthesis with protected place/fact references and weather-aware recommendations.
- Structured evidence, resolved context, missing fields, required tools, and node-level traces.
- Bounded worker pool, input validation, persistence abstraction, health checks, and regression tests.
- Firebase ID-token verification, per-instance rate limiting, idempotent chat retries, and session ownership.
- Cloud Run OIDC authentication for private BE-to-KB calls, retry/cache/circuit breaking, and OpenTelemetry.

For Cloud Run, deploy KB with `--no-allow-unauthenticated`, grant the BE runtime service account
`roles/run.invoker`, then set `KB_AUTH_MODE=google_oidc`. Deploy BE with
`--allow-unauthenticated` so Firebase tokens can reach the app-level verifier. Cloud Run services
must use their attached service identities through ADC instead of a JSON key file.
The included rate limiter is per BE instance; use Cloud Armor or an API gateway for a global
production quota across multiple Cloud Run instances.
