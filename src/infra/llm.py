from __future__ import annotations

import json
import re
from typing import Any, cast

from loguru import logger

from src.config import Settings
from src.core_ai.nextrip_agent.answer_generation import fact_display_text, fact_value_text
from src.core_ai.nextrip_agent.conversation import ConversationResolution
from src.core_ai.nextrip_agent.planning import ItineraryPlan, ItineraryPlanDraft
from src.shared.logging import safe_text
from src.shared.telemetry import record_llm_usage, span


SYSTEM_INSTRUCTION = """You are the grounded answer synthesizer for NexTripAI.
Answer in natural Vietnamese using only the supplied GraphRAG and weather context.
Never invent places, addresses, prices, opening hours, ratings, reasons, or sources.
Place, city, and fact values are protected reference tokens such as [[PLACE_1]], [[CITY_1]], and [[FACT_1]].
Use every PLACE reference at least once and every FACT reference exactly once. For a single-place answer, mention the PLACE once in the introduction or heading and do not append it to every fact bullet. CITY references are optional. Never alter or explain a token.
Never create a reference token. Use only reference tokens that exist verbatim in the supplied JSON.
When verified_facts is empty, do not emit any FACT reference.
If the context does not support a detail, omit it.
For recommendations, keep the retrieved order and explain only relationships present in matched_paths.
When structured_itinerary is present, follow its days, slots, times, and place
references exactly. Never reschedule, add, or remove a place. When it is absent,
do not invent a timed itinerary; present retrieved places as candidates only.
When weather_assessment is present, include its forecast and suitability advice in the same
answer, format dates naturally as DD/MM/YYYY, then recommend suitable retrieved places.
Connect weather to a place only when its supplied entity_type or category supports
that conclusion. Do not invent whether a place is indoors, outdoors, open, or closed.
Treat geographic relationships strictly: LOCATED_IN supports "nằm ở" or "thuộc";
NEAR_AREA supports only "gần"; MENTIONS_GEO_AREA does not prove location and
must be described as an unverified mention or omitted. Never upgrade a weaker
relationship into LOCATED_IN.
Retrieved references may represent places, dishes, activities, cities, or geographic areas.
For an entity_detail with several verified facts, write a short overview followed
by readable bullets for useful details such as address, cuisine, signature dishes,
opening hours, rating, price, amenities, and suitability. Integrate every FACT
reference into a labelled sentence or bullet; never append bare FACT references.
Keep the answer concise and useful. Do not mention internal retrieval, JSON, nodes, or scores.
Start with the useful answer directly; do not say that you only have data or lack enough
information when grounded places are available.
Conversation memory is only for continuity, preferences, and resolving user intent.
Never treat a previous assistant answer as verified factual evidence. Current facts must
come from verified_facts, retrieved_places, matched_graph_paths, or weather_assessment.
"""

CONVERSATION_SYSTEM_INSTRUCTION = """You are NexTripAI's conversation contextualizer.
Read the ordered transcript and decide how the current user turn should be handled.

Return route="conversation" only when the user asks about the conversation itself and
the answer is fully contained in the supplied transcript, for example what they
previously asked or what the assistant previously answered. In that case, answer
faithfully from the transcript and do not add travel facts.

Return route="travel" for every request that needs travel knowledge, recommendations,
weather, or current external facts. Rewrite it as one self-contained standalone_message,
resolving pronouns, omitted subjects, preferences, places, dates, and entities from the
transcript. Preserve the user's intent and language. Never invent missing information.
If a reference is genuinely ambiguous, keep the original wording instead of guessing.

Write a compact Vietnamese rolling summary of durable user preferences, decisions,
travel constraints, and the conversation's important outcomes. Do not include secrets,
implementation details, or unsupported facts.
"""

PLANNING_SYSTEM_INSTRUCTION = """You are NexTripAI's grounded itinerary planning agent.
Return only data matching the supplied schema. Select only place_id values that appear
in candidates. Never invent a place, address, opening hour, travel time, or weather fact.
Keep every stop inside the requested city and preserve the requested number of days.
Create a comfortable route with 4-6 stops per full day when candidates permit:
activities, lunch/dinner, a cafe or rest break, and hotel check-in/check-out when an
overnight stay is requested. Do not treat a hotel as a sightseeing activity.
Use opening hours when present. Avoid overlapping times. When weather is unsuitable,
prefer candidates explicitly marked indoor and do not choose candidates explicitly
marked outdoor-only. Use current coordinates and candidate coordinates to prefer a
nearby first stop, but do not estimate exact travel times. Keep rationales concise and
refer only to supplied candidate or weather attributes.
"""


class GeminiConversationContextualizer:
    def __init__(self, app_settings: Settings):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Missing google-genai. Install BE dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        if not app_settings.google_api_key:
            raise RuntimeError(
                "Conversation contextualization requires GOOGLE_API_KEY."
            )
        self._types = types
        self._model = app_settings.gemini_context_model
        self._thinking_level = app_settings.gemini_thinking_level
        self._summary_max_chars = app_settings.conversation_summary_max_chars
        self._input_cost_per_million = app_settings.gemini_input_cost_per_million_usd
        self._output_cost_per_million = app_settings.gemini_output_cost_per_million_usd
        self._client = genai.Client(
            api_key=app_settings.google_api_key,
            http_options=types.HttpOptions(
                timeout=int(app_settings.conversation_context_timeout_seconds * 1000)
            ),
        )

    def close(self) -> None:
        self._client.close()

    def contextualize(
        self,
        *,
        message: str,
        history: list[dict[str, Any]],
        prior_summary: str | None,
        structured_context: dict[str, Any],
    ) -> ConversationResolution:
        transcript = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or "")[:1600],
            }
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        prompt = {
            "prior_summary": prior_summary,
            "recent_transcript": transcript,
            "structured_trip_context": structured_context,
            "current_user_message": message,
            "summary_max_characters": self._summary_max_chars,
        }
        with span("gemini.contextualize_conversation", model=self._model):
            response = self._client.models.generate_content(
                model=self._model,
                contents=json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
                config=self._types.GenerateContentConfig(
                    system_instruction=CONVERSATION_SYSTEM_INSTRUCTION,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=ConversationResolution,
                    thinking_config=self._types.ThinkingConfig(
                        thinking_level=self._thinking_level,
                    ),
                ),
            )
        parsed = response.parsed
        resolution = (
            cast(ConversationResolution, parsed)
            if parsed is not None
            else ConversationResolution.model_validate_json(response.text or "{}")
        )
        if resolution.summary:
            resolution.summary = resolution.summary[: self._summary_max_chars]
        usage = response.usage_metadata
        input_tokens = int(usage.prompt_token_count or 0) if usage else 0
        output_tokens = int(usage.candidates_token_count or 0) if usage else 0
        thinking_tokens = int(usage.thoughts_token_count or 0) if usage else 0
        record_llm_usage(
            self._model,
            input_tokens,
            output_tokens,
            thinking_tokens=thinking_tokens,
            input_cost_per_million=self._input_cost_per_million,
            output_cost_per_million=self._output_cost_per_million,
        )
        logger.info(
            "Conversation contextualized model={} route={} input_tokens={} "
            "output_tokens={} thinking_tokens={}",
            self._model,
            resolution.route,
            input_tokens,
            output_tokens,
            thinking_tokens,
        )
        return resolution


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
        self._model = app_settings.gemini_answer_model
        self._planning_model = app_settings.gemini_planning_model
        self._thinking_level = app_settings.gemini_thinking_level
        self._planning_thinking_level = app_settings.gemini_planning_thinking_level
        self._temperature = app_settings.answer_temperature
        self._input_cost_per_million = app_settings.gemini_input_cost_per_million_usd
        self._output_cost_per_million = app_settings.gemini_output_cost_per_million_usd
        http_options = types.HttpOptions(
            timeout=int(app_settings.gemini_timeout_seconds * 1000)
        )
        if not app_settings.google_api_key:
            raise RuntimeError("Gemini answer generation requires GOOGLE_API_KEY.")
        self._client = genai.Client(
            api_key=app_settings.google_api_key,
            http_options=http_options,
        )

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
        conversation_context: dict[str, Any] | None = None,
    ) -> str:
        return self._generate(
            question=question,
            answer_type=answer_type,
            evidence=evidence,
            facts=facts,
            matched_paths=matched_paths,
            weather=None,
            conversation_context=conversation_context,
        )

    def synthesize(
        self,
        *,
        question: str,
        answer_type: str,
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        matched_paths: list[dict[str, Any]],
        weather: dict[str, Any] | None,
        conversation_context: dict[str, Any] | None = None,
        itinerary: list[dict[str, Any]] | None = None,
    ) -> str:
        return self._generate(
            question=question,
            answer_type=answer_type,
            evidence=evidence,
            facts=facts,
            matched_paths=matched_paths,
            weather=weather,
            conversation_context=conversation_context,
            itinerary=itinerary,
        )

    def plan_itinerary(
        self,
        *,
        question: str,
        city: str,
        duration_days: int,
        duration_nights: int,
        candidates: list[dict[str, Any]],
        weather: list[dict[str, Any]],
        latitude: float | None,
        longitude: float | None,
        personalization_context: dict[str, Any] | None = None,
    ) -> ItineraryPlan:
        prompt = {
            "question": question,
            "city": city,
            "duration_days": duration_days,
            "duration_nights": duration_nights,
            "origin": (
                {"latitude": latitude, "longitude": longitude}
                if latitude is not None and longitude is not None
                else None
            ),
            "weather": weather,
            "personalization": personalization_context or {},
            "candidates": [_planning_candidate(item) for item in candidates],
        }
        with span("gemini.plan_itinerary", model=self._planning_model):
            response = self._client.models.generate_content(
                model=self._planning_model,
                contents=json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
                config=self._types.GenerateContentConfig(
                    system_instruction=PLANNING_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ItineraryPlanDraft,
                    thinking_config=self._types.ThinkingConfig(
                        thinking_level=self._planning_thinking_level,
                    ),
                ),
            )
        parsed = response.parsed
        draft = (
            cast(ItineraryPlanDraft, parsed)
            if parsed is not None
            else ItineraryPlanDraft.model_validate_json(response.text or "{}")
        )
        plan = ItineraryPlan.model_validate(draft.model_dump(mode="python"))
        usage = response.usage_metadata
        input_tokens = int(usage.prompt_token_count or 0) if usage else 0
        output_tokens = int(usage.candidates_token_count or 0) if usage else 0
        thinking_tokens = int(usage.thoughts_token_count or 0) if usage else 0
        record_llm_usage(
            self._planning_model,
            input_tokens,
            output_tokens,
            thinking_tokens=thinking_tokens,
            input_cost_per_million=self._input_cost_per_million,
            output_cost_per_million=self._output_cost_per_million,
        )
        logger.info(
            "Itinerary planned model={} days={} input_tokens={} output_tokens={} thinking_tokens={}",
            self._planning_model,
            len(plan.days),
            input_tokens,
            output_tokens,
            thinking_tokens,
        )
        return plan

    def _generate(
        self,
        *,
        question: str,
        answer_type: str,
        evidence: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        matched_paths: list[dict[str, Any]],
        weather: dict[str, Any] | None,
        conversation_context: dict[str, Any] | None,
        itinerary: list[dict[str, Any]] | None = None,
    ) -> str:
        if itinerary:
            selected_ids = {
                str(slot.get("place_id"))
                for day in itinerary
                for slot in day.get("slots", [])
                if slot.get("place_id")
            }
            evidence = [
                item
                for item in evidence
                if str(item.get("place_id")) in selected_ids
            ]
            facts = [
                item
                for item in facts
                if str(item.get("subject_id")) in selected_ids
            ]
            matched_paths = [
                item
                for item in matched_paths
                if str(item.get("place_id")) in selected_ids
            ]
        context, replacements = _protected_context(
            question=question,
            answer_type=answer_type,
            evidence=evidence,
            facts=facts,
            matched_paths=matched_paths,
            weather=weather,
            conversation_context=conversation_context,
            itinerary=itinerary,
        )
        with span("gemini.generate_answer", model=self._model, answer_type=answer_type):
            response = self._client.models.generate_content(
                model=self._model,
                contents=json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                config=self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=self._temperature,
                    thinking_config=self._types.ThinkingConfig(
                        thinking_level=self._thinking_level,
                    ),
                ),
            )
        raw_answer = (response.text or "").strip()
        raw_answer = _ensure_single_entity_reference(
            raw_answer,
            answer_type=answer_type,
            evidence=evidence,
        )
        grounded_answer = _restore_references(raw_answer, replacements)
        usage = response.usage_metadata
        input_tokens = int(usage.prompt_token_count or 0) if usage else 0
        output_tokens = int(usage.candidates_token_count or 0) if usage else 0
        thinking_tokens = int(usage.thoughts_token_count or 0) if usage else 0
        record_llm_usage(
            self._model,
            input_tokens,
            output_tokens,
            thinking_tokens=thinking_tokens,
            input_cost_per_million=self._input_cost_per_million,
            output_cost_per_million=self._output_cost_per_million,
        )
        logger.info(
            "NexTrip answer generated model={} input_tokens={} output_tokens={} "
            "thinking_tokens={} answer_len={}",
            self._model,
            input_tokens,
            output_tokens,
            thinking_tokens,
            len(grounded_answer),
        )
        logger.debug(
            "NexTrip grounded answer preview={}", safe_text(grounded_answer, 500)
        )
        return grounded_answer


def _protected_context(
    *,
    question: str,
    answer_type: str,
    evidence: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    matched_paths: list[dict[str, Any]],
    weather: dict[str, Any] | None = None,
    conversation_context: dict[str, Any] | None = None,
    itinerary: list[dict[str, Any]] | None = None,
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
        protected_places.append(
            {
                "reference": reference,
                "city": city_reference,
                "entity_type": place.get("entity_type"),
                "category": place.get("category"),
                "attributes": place.get("attributes") or {},
            }
        )

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
        value = fact_value_text(fact["value"])
        unit = fact.get("unit")
        replacements[reference] = fact_display_text(value, unit)
        protected_facts.append(
            {
                "reference": reference,
                "subject": place_references.get(str(fact.get("subject_id"))),
                "predicate": fact.get("predicate"),
                "entity_type": fact.get("entity_type"),
            }
        )

    protected_paths = []
    for path in matched_paths:
        place_id = str(path.get("place_id") or "")
        protected_paths.append(
            {
                "subject": place_references.get(place_id),
                "concepts": [
                    node for node in path.get("nodes", []) if node != place_id
                ],
                "relationships": path.get("relationships", []),
            }
        )

    protected_itinerary = []
    for day in itinerary or []:
        slots = []
        for slot in day.get("slots", []):
            place_reference = place_references.get(str(slot.get("place_id") or ""))
            if place_reference is None:
                continue
            slots.append(
                {
                    "order": slot.get("order"),
                    "start_time": slot.get("start_time"),
                    "end_time": slot.get("end_time"),
                    "place": place_reference,
                    "entity_type": slot.get("entity_type"),
                    "role": slot.get("role"),
                    "rationale": slot.get("rationale"),
                }
            )
        if slots:
            protected_itinerary.append({"day": day.get("day"), "slots": slots})

    return (
        {
            "question": protected_question,
            "answer_type": answer_type,
            "retrieved_places": protected_places,
            "verified_facts": protected_facts,
            "matched_graph_paths": protected_paths,
            "structured_itinerary": protected_itinerary,
            "weather_assessment": weather,
            "conversation_memory": conversation_context,
        },
        replacements,
    )


def _planning_candidate(item: dict[str, Any]) -> dict[str, Any]:
    attributes = item.get("attributes") or {}
    supported_attributes = {
        key: attributes[key]
        for key in (
            "address",
            "lat",
            "lng",
            "latitude",
            "longitude",
            "opening_hours_open",
            "opening_hours_close",
            "duration_recommendation",
            "indoor",
            "is_indoor",
            "weather_suitable",
            "price_per_person_min",
            "price_per_person_max",
            "rating",
        )
        if key in attributes
    }
    return {
        "place_id": item.get("place_id"),
        "name": item.get("name"),
        "city": item.get("city"),
        "entity_type": item.get("entity_type"),
        "category": item.get("category"),
        "score": item.get("score"),
        "distance_km": item.get("distance_km"),
        "attributes": supported_attributes,
    }


def _ensure_single_entity_reference(
    answer: str,
    *,
    answer_type: str,
    evidence: list[dict[str, Any]],
) -> str:
    reference = "[[PLACE_1]]"
    if not answer:
        return answer
    if answer_type != "entity_detail" or len(evidence) != 1:
        return answer
    if reference not in answer:
        return f"{reference}\n{answer}"
    return answer


def _restore_references(answer: str, replacements: dict[str, str]) -> str:
    if not answer:
        raise RuntimeError("Gemini returned an empty answer")
    missing = []
    for reference in replacements:
        count = answer.count(reference)
        if reference.startswith("[[CITY_"):
            continue
        if reference.startswith("[[PLACE_") and count < 1:
            missing.append(reference)
        elif reference.startswith("[[FACT_") and count != 1:
            missing.append(reference)
    if missing:
        raise RuntimeError(f"Gemini violated grounded reference contract: {missing}")
    place_references = [
        reference
        for reference in replacements
        if reference.startswith("[[PLACE_")
    ]
    if len(place_references) == 1:
        reference = place_references[0]
        before, separator, after = answer.partition(reference)
        answer = before + separator + after.replace(reference, "")
    for reference, value in replacements.items():
        answer = answer.replace(reference, value)
    for value in set(replacements.values()):
        if not value.strip():
            continue
        answer = re.sub(
            rf"(?<!\w){re.escape(value)}(?:\s+{re.escape(value)})+(?!\w)",
            value,
            answer,
            flags=re.IGNORECASE,
        )
    unresolved = re.findall(r"\[\[[A-Z][A-Z0-9_]*\]\]", answer)
    if unresolved:
        raise RuntimeError(f"Gemini emitted unknown grounded references: {unresolved}")
    return answer


def create_answer_generator(app_settings: Settings) -> GeminiAnswerGenerator | None:
    if app_settings.answer_generation_mode == "template":
        return None
    if app_settings.answer_generation_mode != "gemini":
        raise ValueError("ANSWER_GENERATION_MODE must be either 'template' or 'gemini'")
    return GeminiAnswerGenerator(app_settings)


def create_conversation_contextualizer(
    app_settings: Settings,
) -> GeminiConversationContextualizer | None:
    if not app_settings.conversation_context_enabled:
        return None
    if not app_settings.google_api_key:
        logger.warning(
            "Conversation contextualization disabled because GOOGLE_API_KEY is not set"
        )
        return None
    return GeminiConversationContextualizer(app_settings)
