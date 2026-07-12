from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from loguru import logger

from src.config import Settings
from src.shared.logging import safe_text


SYSTEM_INSTRUCTION = """You are the grounded answer generator for NexTripAI.
Answer in natural Vietnamese using only the supplied GraphRAG context.
Never invent places, addresses, prices, opening hours, ratings, reasons, or sources.
Place, city, and fact values are protected reference tokens such as [[PLACE_1]], [[CITY_1]], and [[FACT_1]].
Copy every PLACE and FACT reference exactly once into the answer. CITY references are optional. Never alter or explain a token.
Never create a reference token. Use only reference tokens that exist verbatim in the supplied JSON.
When verified_facts is empty, do not emit any FACT reference.
If the context does not support a detail, omit it.
For recommendations, keep the retrieved order and explain only relationships present in matched_paths.
Keep the answer concise and useful. Do not mention internal retrieval, JSON, nodes, or scores.
"""


class GeminiAnswerGenerator:
    def __init__(self, app_settings: Settings):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Missing google-genai. Install BE dependencies with: pip install -r requirements.txt"
            ) from exc

        self._types = types
        self._model = app_settings.gemini_model
        self._temperature = app_settings.answer_temperature
        if app_settings.google_genai_use_vertexai:
            if not app_settings.google_cloud_project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex AI")
            client_credentials = None
            if app_settings.google_application_credentials:
                credentials_path = Path(app_settings.google_application_credentials)
                if not credentials_path.exists():
                    raise RuntimeError(f"Google credentials file does not exist: {credentials_path}")
                from google.oauth2 import service_account

                client_credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            self._client = genai.Client(
                vertexai=True,
                project=app_settings.google_cloud_project,
                location=app_settings.google_cloud_location,
                credentials=client_credentials,
            )
        elif app_settings.google_api_key:
            self._client = genai.Client(api_key=app_settings.google_api_key)
        else:
            raise RuntimeError("Gemini answer generation is enabled but authentication is missing")

    def close(self) -> None:
        self._client.close()

    def generate(
        self,
        *,
        question: str,
        answer_type: str,
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        matched_paths: list[dict[str, Any]],
    ) -> str:
        context, replacements = _protected_context(
            question=question,
            answer_type=answer_type,
            evidence=evidence,
            facts=facts,
            matched_paths=matched_paths,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self._temperature,
            ),
        )
        raw_answer = (response.text or "").strip()
        grounded_answer = _restore_references(raw_answer, replacements)
        logger.info(
            "NexTrip answer LLM output raw={} grounded={}",
            safe_text(raw_answer, 4000),
            safe_text(grounded_answer, 4000),
        )
        return grounded_answer


def _protected_context(
    *,
    question: str,
    answer_type: str,
    evidence: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    matched_paths: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    replacements: dict[str, str] = {}
    place_references: dict[str, str] = {}
    city_references: dict[str, str] = {}
    protected_places = []
    for index, place in enumerate(evidence, start=1):
        reference = f"[[PLACE_{index}]]"
        place_id = str(place["place_id"])
        city = place.get("city")
        city_reference = None
        if city:
            city_name = str(city)
            city_reference = city_references.get(city_name)
            if city_reference is None:
                city_reference = f"[[CITY_{len(city_references) + 1}]]"
                city_references[city_name] = city_reference
                replacements[city_reference] = city_name
        replacements[reference] = str(place.get("name") or place_id)
        place_references[place_id] = reference
        protected_places.append({
            "reference": reference,
            "city": city_reference,
            "entity_type": place.get("entity_type"),
            "category": place.get("category"),
        })

    protected_question = question
    named_references = sorted(
        (
            (value, reference)
            for reference, value in replacements.items()
            if reference.startswith("[[PLACE_")
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for name, reference in named_references:
        protected_question = re.sub(
            re.escape(name),
            reference,
            protected_question,
            flags=re.IGNORECASE,
        )

    protected_facts = []
    for index, fact in enumerate(facts, start=1):
        reference = f"[[FACT_{index}]]"
        value = str(fact["value"])
        unit = fact.get("unit")
        replacements[reference] = f"{value} {unit}" if unit else value
        protected_facts.append({
            "reference": reference,
            "subject": place_references.get(str(fact.get("subject_id"))),
            "predicate": fact.get("predicate"),
            "entity_type": fact.get("entity_type"),
        })

    protected_paths = []
    for path in matched_paths:
        place_id = str(path.get("place_id") or "")
        protected_paths.append({
            "subject": place_references.get(place_id),
            "concepts": [
                node for node in path.get("nodes", [])
                if node != place_id
            ],
            "relationships": path.get("relationships", []),
        })

    return (
        {
            "question": protected_question,
            "answer_type": answer_type,
            "retrieved_places": protected_places,
            "verified_facts": protected_facts,
            "matched_graph_paths": protected_paths,
        },
        replacements,
    )


def _restore_references(answer: str, replacements: dict[str, str]) -> str:
    if not answer:
        raise RuntimeError("Gemini returned an empty answer")
    missing = [
        reference
        for reference in replacements
        if not reference.startswith("[[CITY_") and answer.count(reference) != 1
    ]
    if missing:
        raise RuntimeError(f"Gemini violated grounded reference contract: {missing}")
    for reference, value in replacements.items():
        answer = answer.replace(reference, value)
    unresolved = re.findall(r"\[\[[A-Z][A-Z0-9_]*\]\]", answer)
    if unresolved:
        raise RuntimeError(f"Gemini emitted unknown grounded references: {unresolved}")
    return answer


def create_answer_generator(app_settings: Settings) -> GeminiAnswerGenerator | None:
    if app_settings.answer_generation_mode == "template":
        return None
    if app_settings.answer_generation_mode != "gemini":
        raise ValueError(
            "ANSWER_GENERATION_MODE must be either 'template' or 'gemini'"
        )
    return GeminiAnswerGenerator(app_settings)
