# NexTripAI-BE

Backend service for NexTripAI. This repo owns the API layer, AI agent orchestration, user/session data, and integration with the Knowledge Base, Maps, Weather, and Frontend.

## Workflow Documentation

Read this before implementation:

- Repo workflow: [docs/WORKFLOW.md](docs/WORKFLOW.md)
- System workflow: [../docs/WORKFLOW.md](../docs/WORKFLOW.md)
- Repo structure guide: [../docs/REPO_STRUCTURE.md](../docs/REPO_STRUCTURE.md)
- Step-by-step roadmap: [../docs/IMPLEMENTATION_STEPS.md](../docs/IMPLEMENTATION_STEPS.md)

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
