from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
import re
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

from loguru import logger

from src.apis.domains.chat.schemas import (
    ChatRequest,
    ChatResponse,
    Clarification,
    ClarificationOption,
    EvidenceItem,
    PlanningOutcome,
)
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.core_ai.nextrip_agent.current_data import (
    SupportsCurrentData,
    attach_itinerary_transport,
    enrich_current_data,
)
from src.core_ai.nextrip_agent.constants import (
    is_typed_kb_version,
    supports_structured_conversation_context,
)
from src.core_ai.nextrip_agent.conversation import (
    ConversationContext,
    ResolvedTurn,
    SupportsConversationContextualization,
    answer_memory_context,
    SUPPORTED_CITIES,
    resolve_conversation_context,
    resolve_turn,
)
from src.core_ai.nextrip_agent.orchestrator import TravelOrchestrator
from src.core_ai.nextrip_agent.planning import is_itinerary_request
from src.core_ai.nextrip_agent.schemas import AgentResult
from src.core_ai.nextrip_agent.synthesizer import SynthesisResult, synthesize_answer
from src.core_ai.nextrip_agent.trip_plan import (
    ActiveTripPlan,
    PlanChange,
    PlanMutation,
    PlanOperation,
    active_plan_from_value,
    apply_plan_mutation,
    build_active_trip_plan,
    compact_active_plan_context,
    invalidate_adjacent_routes,
    locate_target_slot,
    rank_nearby_candidates,
    resolve_plan_mutation,
)
from src.core_ai.personalization.service import compile_personalization_context
from src.core_ai.personalization.models import PreferenceEvent
from src.infra.kb_client import KbClient
from src.infra.chat_store import ChatStore, TripPlanRevisionConflictError
from src.infra.user_profile_store import UserProfileStore
from src.infra.weather import OpenMeteoWeatherClient

DEFAULT_TOP_K = 5
TYPED_QUERY_RESULT_CEILING = 20
_VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


class KnowledgeBaseUnavailableError(RuntimeError):
    pass


class KnowledgeBaseVersionMismatchError(RuntimeError):
    pass


class ConversationKnowledgeBaseVersionError(RuntimeError):
    pass


def _build_clarification(
    missing_fields: list[str],
    *,
    required_tools: list[str],
) -> Clarification | None:
    if "city" in missing_fields:
        weather_prompt = bool({"weather", "weather_forecast"} & set(required_tools))
        return Clarification(
            field="city",
            prompt=(
                "Bạn muốn xem thời tiết ở thành phố nào?"
                if weather_prompt
                else "Bạn muốn đi Quy Nhơn, Đà Nẵng hay xem gợi ý ở cả hai thành phố?"
            ),
            options=[
                ClarificationOption(label=city, value=city)
                for city in SUPPORTED_CITIES.values()
            ]
            + [ClarificationOption(label="Cả hai thành phố", value="all")],
        )
    if "query_constraints" in missing_fields:
        return Clarification(
            field="query_constraints",
            prompt="Bạn muốn ưu tiên thành phố nào, hay tìm trên cả hai?",
            options=[
                ClarificationOption(label=city, value=city)
                for city in SUPPORTED_CITIES.values()
            ]
            + [ClarificationOption(label="Cả hai thành phố", value="all")],
        )
    if "near_reference" in missing_fields:
        return Clarification(
            field="near_reference",
            prompt="Bạn muốn ưu tiên khoảng cách theo mốc nào?",
            options=[
                ClarificationOption(
                    label="Gần trung tâm",
                    value="Ưu tiên cách trung tâm thành phố không quá 3 km.",
                ),
                ClarificationOption(
                    label="Gần biển",
                    value="Ưu tiên cách biển không quá 2 km.",
                ),
                ClarificationOption(
                    label="Không giới hạn",
                    value="Bỏ tiêu chí khoảng cách và tìm trong toàn thành phố.",
                ),
            ],
        )
    if "geo_area" in missing_fields:
        return Clarification(
            field="geo_area",
            prompt="Khu vực này chưa khớp chắc chắn. Bạn muốn mở rộng phạm vi chứ?",
            options=[
                ClarificationOption(
                    label="Toàn thành phố",
                    value="Bỏ giới hạn khu vực và tìm trong toàn thành phố.",
                ),
                ClarificationOption(
                    label="Chỉ vị trí xác minh",
                    value="Chỉ dùng địa điểm có quan hệ vị trí đã xác minh.",
                ),
            ],
        )
    return None


def resolve_top_k(request: ChatRequest) -> int:
    if request.top_k is not None:
        return request.top_k
    if is_typed_kb_version(request.kb_version):
        return TYPED_QUERY_RESULT_CEILING
    return DEFAULT_TOP_K


def _evidence_type_counts(evidence: list[EvidenceItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        key = str(item.entity_type or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _resolve_effective_travel_date(
    *,
    explicit_date: date | None,
    context_date: date | None,
    active_plan: ActiveTripPlan | None,
    message: str,
    now: datetime | None = None,
) -> tuple[date | None, bool]:
    """Resolve one date context for planning, dynamic prices, hours and routes.

    An undated itinerary starts tomorrow so the system can produce a usable
    hotel quote instead of silently omitting accommodation prices. The
    assumption is persisted in plan constraints and exposed by the frontend.
    """

    reference = now or datetime.now(_VIETNAM_TIMEZONE)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=_VIETNAM_TIMEZONE)
    else:
        reference = reference.astimezone(_VIETNAM_TIMEZONE)

    if explicit_date is not None:
        return explicit_date, False
    message_date = _travel_date_from_message(message, reference.date())
    if message_date is not None:
        return message_date, False
    if context_date is not None:
        return context_date, False
    if active_plan is not None and active_plan.start_date is not None:
        return (
            active_plan.start_date,
            bool(active_plan.constraints.get("travel_date_assumed")),
        )
    if not is_itinerary_request(message, {}):
        return None, False
    return reference.date() + timedelta(days=1), True


def _resolve_current_data_travel_date(
    *,
    travel_date: date | None,
    required_tools: list[str],
    now: datetime | None = None,
) -> tuple[date | None, bool]:
    """Use today's operational window for an undated hotel-price request."""

    if travel_date is not None:
        return travel_date, False
    if not {"booking", "live_price"}.intersection(required_tools):
        return None, False

    reference = now or datetime.now(_VIETNAM_TIMEZONE)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=_VIETNAM_TIMEZONE)
    else:
        reference = reference.astimezone(_VIETNAM_TIMEZONE)
    return reference.date(), True


def _travel_date_from_message(message: str, today: date) -> date | None:
    """Parse an explicit Vietnamese numeric date without confusing trip length."""

    match = re.search(
        r"(?<!\d)([0-3]?\d)[/-]([01]?\d)(?:[/-](\d{2}|\d{4}))?(?!\d)",
        message,
    )
    if match is None:
        return None
    day_value = int(match.group(1))
    month_value = int(match.group(2))
    year_text = match.group(3)
    year_value = (
        2000 + int(year_text)
        if year_text is not None and len(year_text) == 2
        else int(year_text or today.year)
    )
    try:
        parsed = date(year_value, month_value, day_value)
    except ValueError:
        return None
    if year_text is None and parsed < today:
        try:
            parsed = parsed.replace(year=parsed.year + 1)
        except ValueError:
            return None
    return parsed


def handle_chat(
    request: ChatRequest,
    kb_client: KbClient,
    answer_generator: SupportsAnswerGeneration | None = None,
    *,
    weather_client: OpenMeteoWeatherClient | None = None,
    chat_store: ChatStore | None = None,
    chat_history_limit: int = 8,
    conversation_contextualizer: SupportsConversationContextualization | None = None,
    user_profile_store: UserProfileStore | None = None,
    current_data_client: SupportsCurrentData | None = None,
) -> ChatResponse:
    started_at = perf_counter()
    user_id = getattr(request, "user_id", None)
    top_k = resolve_top_k(request)
    user_message_id = uuid4().hex
    assistant_message_id = uuid4().hex
    history = _recent_messages(
        chat_store,
        request.session_id,
        chat_history_limit,
        user_id=user_id,
    )
    session_memory = _get_session_memory(
        chat_store,
        request.session_id,
        user_id=user_id,
    )
    active_plan = _get_active_trip_plan(
        chat_store,
        request.session_id,
        user_id=user_id,
    )
    session_kb_version = session_memory.get("kb_version")
    if session_kb_version and session_kb_version != request.kb_version:
        raise ConversationKnowledgeBaseVersionError(
            "This conversation is pinned to Knowledge Base "
            f"{str(session_kb_version).upper()}. Start a new conversation to use "
            f"{request.kb_version.upper()}."
        )
    personalization = {}
    personalization_enabled = False
    if user_profile_store is not None and user_id:
        try:
            profile = user_profile_store.get_profile(user_id)
            personalization_enabled = profile.personalization_enabled
            personalization = compile_personalization_context(profile)
        except Exception as exc:
            logger.warning(
                "Personalization load failed user_id={} error_type={}",
                user_id,
                exc.__class__.__name__,
            )
    context = resolve_conversation_context(
        message=request.message,
        explicit_city=request.city,
        history=history,
        explicit_travel_date=request.travel_date,
    )
    resolved_turn = resolve_turn(
        message=request.message,
        history=history,
        context=context,
        contextualizer=conversation_contextualizer,
        prior_summary=str(session_memory.get("summary") or "") or None,
        active_trip_plan=compact_active_plan_context(active_plan),
    )
    plan_mutation = resolve_plan_mutation(
        request.message,
        active_plan,
        resolved_turn.resolution.plan_mutation,
    )
    if request.expected_plan_revision is not None and _is_mutating_plan_operation(
        plan_mutation.operation
    ):
        plan_mutation = plan_mutation.model_copy(
            update={"expected_revision": request.expected_plan_revision}
        )
    if _is_mutating_plan_operation(plan_mutation.operation):
        _require_expected_plan_revision(
            request.session_id,
            active_plan,
            plan_mutation.expected_revision,
        )
    structured_context = supports_structured_conversation_context(request.kb_version)
    stored_kb_context = session_memory.get("kb_context")
    if not isinstance(stored_kb_context, dict):
        stored_kb_context = None
    if structured_context:
        graph_message = resolved_turn.resolution.standalone_message
        kb_conversation_context = _kb_conversation_context(
            session_memory=session_memory,
            context=context,
            resolved_turn=resolved_turn,
            personalization=personalization,
        )
    else:
        graph_message = resolved_turn.resolution.standalone_message
        kb_conversation_context = stored_kb_context
    effective_city = context.city
    effective_entity_types = request.entity_types
    discovery_target = None
    discovery_anchor = None
    if active_plan is not None and plan_mutation.operation in {
        PlanOperation.ADD_SLOT,
        PlanOperation.REPLACE_SLOT,
        PlanOperation.SUGGEST_NEARBY,
    }:
        discovery_target, discovery_anchor = _plan_discovery_context(
            active_plan,
            plan_mutation,
        )
        target_type = _discovery_entity_type(plan_mutation, discovery_target)
        graph_message = _plan_discovery_message(
            request.message,
            city=active_plan.city,
            entity_type=target_type,
            anchor_name=(discovery_anchor or {}).get("name"),
        )
        effective_city = active_plan.city
        effective_entity_types = [target_type] if target_type else request.entity_types
        kb_conversation_context = {
            "cities": [active_plan.city],
            **({"personalization": personalization} if personalization else {}),
        }
    _save_message(
        chat_store,
        request,
        user_message_id,
        "user",
        request.display_message or request.message,
        city=context.city,
        metadata={
            "context_city_source": context.city_source,
            "travel_date": request.travel_date.isoformat()
            if request.travel_date
            else None,
            "conversation_route": resolved_turn.resolution.route,
        },
    )
    local_plan_response = _handle_local_plan_action(
        request=request,
        active_plan=active_plan,
        mutation=plan_mutation,
        current_data_client=current_data_client,
        chat_store=chat_store,
        assistant_message_id=assistant_message_id,
        context=context,
        resolved_turn=resolved_turn,
        session_memory=session_memory,
    )
    if local_plan_response is not None:
        return local_plan_response
    logger.bind(
        event="chat.turn.started",
        city_resolved=context.city is not None,
        entity_type_count=len(request.entity_types or []),
        top_k=top_k,
        message_length=len(request.message),
        conversation_route=resolved_turn.resolution.route,
    ).info("chat.turn.started")
    if resolved_turn.resolution.route == "conversation":
        answer = str(resolved_turn.resolution.direct_answer or "").strip()
        trace = [context.trace_event(), resolved_turn.trace_event()]
        resolved_context = _resolved_context(context.to_dict(), resolved_turn)
        response = ChatResponse(
            session_id=request.session_id,
            message_id=assistant_message_id,
            answer=answer,
            intent="conversation_memory",
            orchestration_mode="conversation",
            resolved_context=resolved_context,
            kb_version=request.kb_version,
            trace=trace,
        )
        _save_message(
            chat_store,
            request,
            assistant_message_id,
            "assistant",
            answer,
            city=context.city,
            metadata={
                "kb_version": request.kb_version,
                "resolved_context": resolved_context,
                "trace": trace,
            },
        )
        _save_session_memory(
            chat_store,
            request.session_id,
            existing=session_memory,
            summary=resolved_turn.resolution.summary,
            kb_version=request.kb_version,
            user_id=user_id,
        )
        logger.info(
            "Chat turn end session_id={} route=conversation answer_len={} elapsed_ms={}",
            request.session_id,
            len(answer),
            int((perf_counter() - started_at) * 1000),
        )
        return response

    effective_travel_date, travel_date_assumed = _resolve_effective_travel_date(
        explicit_date=request.travel_date,
        context_date=context.travel_date,
        active_plan=active_plan,
        message=graph_message,
    )
    logger.bind(
        event="itinerary.context.resolved",
        itinerary_requested=is_itinerary_request(graph_message, {}),
        city_resolved=effective_city is not None,
        travel_date=effective_travel_date.isoformat()
        if effective_travel_date
        else None,
        travel_date_assumed=travel_date_assumed,
        active_plan_present=active_plan is not None,
    ).info("itinerary.context.resolved")
    orchestration = TravelOrchestrator(
        kb_client,
        weather_client,
        planning_agent=answer_generator,
        current_data_client=current_data_client,
    ).run(
        message=graph_message,
        session_id=request.session_id,
        city=effective_city,
        entity_types=effective_entity_types,
        top_k=top_k,
        kb_version=request.kb_version,
        travel_date=effective_travel_date,
        include_weather=request.include_weather,
        latitude=request.latitude,
        longitude=request.longitude,
        conversation_context=kb_conversation_context,
    )
    current_data_travel_date, current_data_date_assumed = (
        _resolve_current_data_travel_date(
            travel_date=effective_travel_date,
            required_tools=orchestration.graph.required_tools,
        )
    )
    if current_data_date_assumed:
        logger.bind(
            event="itinerary.current_data.date_assumed",
            travel_date=current_data_travel_date.isoformat(),
            required_tools=orchestration.graph.required_tools,
        ).info("Current Data hotel date assumed")
    effective_travel_date = current_data_travel_date
    travel_date_assumed = travel_date_assumed or current_data_date_assumed
    active_constraints = active_plan.constraints if active_plan is not None else {}
    agent_result, current_data_trace = enrich_current_data(
        orchestration.graph,
        current_data_client,
        travel_date=effective_travel_date,
        adults=plan_mutation.adults
        or int(
            active_constraints.get("adults")
            or active_constraints.get("party_size")
            or 2
        ),
        children=plan_mutation.children
        if plan_mutation.children is not None
        else int(active_constraints.get("children") or 0),
        rooms=plan_mutation.rooms or int(active_constraints.get("rooms") or 1),
    )
    spatial_candidates = _load_spatial_candidates(
        kb_client,
        active_plan=active_plan,
        mutation=plan_mutation,
        target=discovery_target,
        anchor=discovery_anchor,
    )
    if spatial_candidates is not None:
        spatial_graph, spatial_trace = enrich_current_data(
            AgentResult(
                answer="",
                answer_type="recommendation",
                evidence=spatial_candidates,
                query_plan={
                    "duration_days": active_plan.duration_days
                    if active_plan is not None
                    else 1
                },
            ),
            current_data_client,
            travel_date=effective_travel_date,
            include_traffic=False,
            adults=plan_mutation.adults
            or int(
                active_constraints.get("adults")
                or active_constraints.get("party_size")
                or 2
            ),
            children=plan_mutation.children
            if plan_mutation.children is not None
            else int(active_constraints.get("children") or 0),
            rooms=plan_mutation.rooms or int(active_constraints.get("rooms") or 1),
        )
        if plan_mutation.operation is PlanOperation.SUGGEST_NEARBY:
            candidate_evidence = spatial_graph.evidence
        else:
            candidate_evidence = _merge_evidence_rows(
                spatial_graph.evidence,
                agent_result.evidence,
            )
        agent_result = agent_result.model_copy(update={"evidence": candidate_evidence})
        current_data_trace = {
            **current_data_trace,
            "nearby": spatial_trace,
            "nearby_candidate_count": len(spatial_graph.evidence),
        }
    weather = orchestration.weather
    if agent_result.error and agent_result.error.get("code") == "kb_version_mismatch":
        raise KnowledgeBaseVersionMismatchError(agent_result.error["message"])
    if agent_result.error and weather is None:
        raise KnowledgeBaseUnavailableError(agent_result.error["message"])
    response_active_plan: ActiveTripPlan | None = None
    plan_change: PlanChange | None = None
    replan_unchanged = False
    nearby_suggestion_rows: list[dict] = []
    if active_plan is not None and plan_mutation.operation in {
        PlanOperation.ADD_SLOT,
        PlanOperation.REPLACE_SLOT,
        PlanOperation.SUGGEST_NEARBY,
    }:
        used_ids = {
            str(slot.get("place_id") or "")
            for day_item in active_plan.itinerary
            for slot in day_item.get("slots", [])
        }
        target_type = _discovery_entity_type(plan_mutation, discovery_target)
        ranked = rank_nearby_candidates(
            agent_result.evidence,
            anchor=discovery_anchor,
            city=active_plan.city,
            entity_type=target_type,
            excluded_place_ids=used_ids,
        )
        if plan_mutation.operation is PlanOperation.SUGGEST_NEARBY:
            nearby_suggestion_rows = ranked[:5]
            agent_result = agent_result.model_copy(
                update={
                    "evidence": nearby_suggestion_rows,
                    "answer_type": "recommendation",
                    "itinerary": [],
                }
            )
            response_active_plan = active_plan
        else:
            try:
                days, selected_places, plan_change = apply_plan_mutation(
                    active_plan,
                    plan_mutation,
                    ranked,
                )
            except ValueError:
                nearby_suggestion_rows = ranked[:5]
                response_active_plan = active_plan
                plan_change = PlanChange(
                    operation=plan_mutation.operation,
                    previous_revision=active_plan.revision,
                    revision=active_plan.revision,
                    message="replacement_not_applied",
                )
                agent_result = agent_result.model_copy(
                    update={
                        "evidence": nearby_suggestion_rows,
                        "answer_type": "recommendation",
                        "itinerary": [],
                        "warnings": [
                            *agent_result.warnings,
                            "plan_replacement_not_applied",
                        ],
                    }
                )
                days = []
                selected_places = []
            if response_active_plan is None:
                days = invalidate_adjacent_routes(days, plan_change.changed_slot_ids)
                days, _, route_failures = attach_itinerary_transport(
                    days,
                    current_data_client,
                    start_date=active_plan.start_date,
                )
                evidence_for_revision = [*selected_places, *ranked]
                response_active_plan = build_active_trip_plan(
                    itinerary=days,
                    evidence=evidence_for_revision,
                    city=active_plan.city,
                    start_date=active_plan.start_date,
                    duration_days=active_plan.duration_days,
                    operation=plan_mutation.operation,
                    previous=active_plan,
                    mutation=plan_mutation,
                )
                plan_change = plan_change.model_copy(
                    update={"revision": response_active_plan.revision}
                )
                _persist_active_trip_plan(
                    chat_store,
                    request.session_id,
                    response_active_plan,
                    expected_revision=active_plan.revision,
                    user_id=user_id,
                )
                agent_result = agent_result.model_copy(
                    update={
                        "evidence": response_active_plan.selected_places,
                        "itinerary": response_active_plan.itinerary,
                        "answer_type": "itinerary_planning",
                        "warnings": [
                            *agent_result.warnings,
                            *(["plan_route_partial"] if route_failures else []),
                        ],
                    }
                )
    elif agent_result.itinerary and (
        active_plan is None
        or plan_mutation.operation
        in {PlanOperation.REPLAN_ALL, PlanOperation.REPLAN_DAY}
        or (
            active_plan is not None
            and plan_mutation.operation is PlanOperation.NONE
            and orchestration.plan.run_planning
        )
    ):
        operation = PlanOperation.CREATE
        itinerary_for_revision = agent_result.itinerary
        evidence_for_revision = agent_result.evidence
        preserved_slot_ids: list[str] = []
        target_day: int | None = None
        if active_plan is not None:
            operation = plan_mutation.operation
            if operation is PlanOperation.REPLAN_DAY:
                target_day = plan_mutation.target_day
                if target_day is None:
                    target_day = 1
                itinerary_for_revision = _merge_replanned_day(
                    active_plan,
                    agent_result.itinerary,
                    target_day=target_day,
                )
                evidence_for_revision = _merge_evidence_rows(
                    active_plan.selected_places,
                    agent_result.evidence,
                )
                changed_ids = _slot_ids_for_day(
                    itinerary_for_revision,
                    target_day,
                )
                itinerary_for_revision = invalidate_adjacent_routes(
                    itinerary_for_revision,
                    changed_ids,
                )
                itinerary_for_revision, _, route_failures = attach_itinerary_transport(
                    itinerary_for_revision,
                    current_data_client,
                    start_date=active_plan.start_date,
                )
                if route_failures:
                    agent_result = agent_result.model_copy(
                        update={
                            "warnings": [
                                *agent_result.warnings,
                                "plan_route_partial",
                            ]
                        }
                    )
                preserved_slot_ids = [
                    slot_id
                    for day_item in active_plan.itinerary
                    if int(day_item.get("day") or 0) != target_day
                    for slot_id in _slot_ids_for_day(
                        active_plan.itinerary,
                        int(day_item.get("day") or 0),
                    )
                ]
            else:
                operation = PlanOperation.REPLAN_ALL
        if (
            active_plan is not None
            and operation in {PlanOperation.REPLAN_ALL, PlanOperation.REPLAN_DAY}
            and plan_mutation.operation
            in {PlanOperation.REPLAN_ALL, PlanOperation.REPLAN_DAY}
            and _itinerary_place_signature(itinerary_for_revision)
            == _itinerary_place_signature(active_plan.itinerary)
        ):
            replan_unchanged = True
            response_active_plan = active_plan
            plan_change = PlanChange(
                operation=operation,
                previous_revision=active_plan.revision,
                revision=active_plan.revision,
                preserved_slot_ids=_all_slot_ids(active_plan),
                message="replan_not_applied",
            )
            agent_result = agent_result.model_copy(
                update={
                    "evidence": active_plan.selected_places,
                    "itinerary": active_plan.itinerary,
                    "warnings": list(
                        dict.fromkeys(
                            [*agent_result.warnings, "plan_replan_unchanged"]
                        )
                    ),
                }
            )
        else:
            response_active_plan = build_active_trip_plan(
                itinerary=itinerary_for_revision,
                evidence=evidence_for_revision,
                city=effective_city or _itinerary_city(agent_result.itinerary),
                start_date=effective_travel_date,
                duration_days=(
                    active_plan.duration_days
                    if active_plan is not None
                    else _itinerary_duration(agent_result.itinerary)
                ),
                operation=operation,
                previous=active_plan,
                mutation=plan_mutation,
                travel_date_assumed=travel_date_assumed,
            )
            changed_slot_ids = (
                _slot_ids_for_day(response_active_plan.itinerary, target_day or 1)
                if active_plan is not None and operation is PlanOperation.REPLAN_DAY
                else _all_slot_ids(response_active_plan)
            )
            plan_change = PlanChange(
                operation=operation,
                previous_revision=active_plan.revision if active_plan else None,
                revision=response_active_plan.revision,
                changed_slot_ids=changed_slot_ids,
                preserved_slot_ids=preserved_slot_ids,
                message=(
                    "created_plan"
                    if active_plan is None
                    else (
                        f"replanned_day:{target_day}"
                        if operation is PlanOperation.REPLAN_DAY
                        else "replanned_all"
                    )
                ),
            )
            _persist_active_trip_plan(
                chat_store,
                request.session_id,
                response_active_plan,
                expected_revision=active_plan.revision if active_plan else None,
                user_id=user_id,
            )
            agent_result = agent_result.model_copy(
                update={
                    "evidence": response_active_plan.selected_places,
                    "itinerary": response_active_plan.itinerary,
                }
            )
    elif (
        active_plan is not None
        and plan_mutation.operation is PlanOperation.NONE
        and agent_result.itinerary
    ):
        # A non-planning follow-up may still receive an itinerary-shaped KB
        # payload. Returning that raw schedule would display an unversioned
        # "ghost" plan beside the persisted active revision, without the cost
        # annotations added by ``build_active_trip_plan``. Keep the unrelated
        # response independent and leave the active plan untouched.
        response_active_plan = active_plan
        agent_result = agent_result.model_copy(update={"itinerary": []})
    planning_unavailable = bool(
        orchestration.plan.run_planning and not agent_result.itinerary
    )
    if replan_unchanged and response_active_plan is not None:
        synthesis = SynthesisResult(
            answer=(
                "Mình đã thử sắp xếp lại nhưng phương án phù hợp nhất vẫn giống "
                "lịch trình hiện tại, nên chưa tạo revision mới. Bạn có thể nêu "
                "tiêu chí muốn thay đổi, chẳng hạn thêm một địa điểm, đổi loại "
                "địa điểm hoặc đổi khung giờ."
            ),
            unresolved_tools=[],
            trace={
                "node": "answer_synthesizer",
                "status": "completed",
                "generator": "plan_noop_guard",
                "reason": "replan_unchanged",
                "sources": [],
            },
        )
    elif planning_unavailable:
        warnings = list(
            dict.fromkeys([*agent_result.warnings, "planning_unavailable"])
        )
        if active_plan is not None:
            # A failed replan must never erase a valid active revision. Return
            # the last committed plan unchanged so the user can keep editing it.
            response_active_plan = active_plan
            agent_result = agent_result.model_copy(
                update={
                    "answer_type": "itinerary_planning",
                    "evidence": active_plan.selected_places,
                    "itinerary": active_plan.itinerary,
                    "warnings": warnings,
                }
            )
        else:
            # Do not present a large candidate dump as if it were an itinerary.
            # Candidates remain an internal planning input; a failed plan is an
            # explicit partial outcome with no public pseudo-schedule.
            agent_result = agent_result.model_copy(
                update={
                    "answer_type": "itinerary_planning",
                    "evidence": [],
                    "facts": [],
                    "matched_paths": [],
                    "itinerary": [],
                    "warnings": warnings,
                }
            )
    evidence = [EvidenceItem.model_validate(item) for item in agent_result.evidence]
    nearby_suggestions = [
        EvidenceItem.model_validate(item) for item in nearby_suggestion_rows
    ]
    answer_context = (
        answer_memory_context(
            history=history,
            resolved_turn=resolved_turn,
        )
        or {}
    )
    if personalization:
        answer_context["personalization"] = personalization
    if response_active_plan is not None and response_active_plan.budget_summary:
        answer_context["budget_summary"] = (
            response_active_plan.budget_summary.model_dump(mode="json")
        )
    if response_active_plan is not None and response_active_plan.start_date is not None:
        answer_context["trip_context"] = {
            "start_date": response_active_plan.start_date.isoformat(),
            "start_date_assumed": bool(
                response_active_plan.constraints.get("travel_date_assumed")
            ),
            "duration_days": response_active_plan.duration_days,
            "party_size": response_active_plan.constraints.get("party_size", 2),
            "rooms": response_active_plan.constraints.get("rooms", 1),
        }
    if planning_unavailable:
        logger.bind(
            event="itinerary.response.guard",
            planning_status=orchestration.planning_trace.get("status"),
            reason_code=orchestration.planning_trace.get("reason"),
            failure_reason=orchestration.planning_trace.get("failure_reason"),
            evidence_count=len(agent_result.evidence),
            itinerary_days=len(agent_result.itinerary),
        ).warning("itinerary.response.guard")
        synthesis = SynthesisResult(
            answer=_planning_unavailable_answer(
                response_active_plan,
                reason=str(orchestration.planning_trace.get("reason") or "") or None,
            ),
            unresolved_tools=[],
            trace={
                "node": "answer_synthesizer",
                "status": "fallback",
                "generator": "planning_guard",
                "reason": orchestration.planning_trace.get("reason")
                or "planning_unavailable",
                "sources": [],
            },
        )
    else:
        synthesis = synthesize_answer(
            question=request.message,
            kb_version=request.kb_version,
            graph=agent_result,
            graph_used=orchestration.plan.run_graph,
            weather=weather,
            weather_requested=orchestration.plan.run_weather,
            weather_trace=orchestration.weather_trace,
            answer_generator=answer_generator,
            conversation_context=answer_context or None,
        )
    answer = synthesis.answer
    trace = [
        context.trace_event(),
        resolved_turn.trace_event(),
        *orchestration.trace,
        *agent_result.trace,
        orchestration.weather_trace,
        orchestration.planning_trace,
        current_data_trace,
        {
            "node": "travel_date_context",
            "status": "assumed" if travel_date_assumed else "resolved",
            "travel_date": effective_travel_date.isoformat()
            if effective_travel_date
            else None,
        },
        synthesis.trace,
    ]
    if response_active_plan is not None:
        trace.append(
            {
                "node": "active_trip_plan",
                "status": "completed",
                "plan_id": response_active_plan.plan_id,
                "revision": response_active_plan.revision,
                "operation": response_active_plan.last_operation.value,
                "budget_status": response_active_plan.budget_summary.status
                if response_active_plan.budget_summary
                else None,
            }
        )
    logger.bind(
        event="chat.turn.completed",
        evidence_count=len(evidence),
        evidence_type_counts=_evidence_type_counts(evidence),
        itinerary_days=len(agent_result.itinerary),
        itinerary_slots=sum(
            len(day_item.get("slots", [])) for day_item in agent_result.itinerary
        ),
        planning_status=orchestration.planning_trace.get("status"),
        answer_length=len(answer),
        elapsed_ms=int((perf_counter() - started_at) * 1000),
    ).info("chat.turn.completed")
    resolved_context = _resolved_context(context.to_dict(), resolved_turn)
    response = ChatResponse(
        session_id=request.session_id,
        message_id=assistant_message_id,
        answer=answer,
        intent=agent_result.answer_type,
        orchestration_mode=orchestration.plan.mode.value,
        resolved_context=resolved_context,
        kb_version=request.kb_version,
        facts=agent_result.facts,
        evidence=evidence,
        recommendations=evidence
        if agent_result.answer_type == "recommendation"
        else [],
        itinerary=agent_result.itinerary,
        warnings=agent_result.warnings,
        missing_fields=agent_result.missing_fields,
        trace=trace,
        query_plan=agent_result.query_plan,
        matched_paths=agent_result.matched_paths,
        constraint_results=agent_result.constraint_results,
        required_tools=synthesis.unresolved_tools,
        clarification=_build_clarification(
            agent_result.missing_fields,
            required_tools=synthesis.unresolved_tools,
        ),
        weather=weather,
        weather_forecast=orchestration.weather_forecast,
        active_trip_plan=response_active_plan,
        plan_change=plan_change,
        budget_summary=(
            response_active_plan.budget_summary if response_active_plan else None
        ),
        nearby_suggestions=nearby_suggestions,
        planning=_planning_outcome(
            orchestration.planning_trace,
            retryable=bool(
                planning_unavailable
                and agent_result.error
                and agent_result.error.get("retryable")
            ),
        ),
    )
    _save_message(
        chat_store,
        request,
        assistant_message_id,
        "assistant",
        answer,
        city=context.city,
        metadata={
            "kb_version": request.kb_version,
            "place_ids": [item.place_id for item in evidence],
            "referenced_entities": _referenced_entities(evidence),
            "weather_suitability": weather.suitability if weather else None,
            "travel_date": effective_travel_date.isoformat()
            if effective_travel_date
            else None,
            "travel_date_assumed": travel_date_assumed,
            "itinerary": agent_result.itinerary,
            "active_trip_plan": response_active_plan.model_dump(mode="json")
            if response_active_plan
            else None,
            "plan_change": plan_change.model_dump(mode="json") if plan_change else None,
            "budget_summary": response_active_plan.budget_summary.model_dump(
                mode="json"
            )
            if response_active_plan and response_active_plan.budget_summary
            else None,
            "nearby_suggestions": [
                item.model_dump(mode="json") for item in nearby_suggestions
            ],
            "warnings": agent_result.warnings,
            "planning": response.planning.model_dump(mode="json"),
            "resolved_context": resolved_context,
            "trace": trace,
        },
    )
    _save_session_memory(
        chat_store,
        request.session_id,
        existing=session_memory,
        summary=resolved_turn.resolution.summary,
        kb_context=agent_result.conversation_context,
        kb_version=request.kb_version,
        user_id=user_id,
    )
    _record_grounded_place_interest(
        user_profile_store,
        user_id=user_id,
        session_id=request.session_id,
        answer_type=str(agent_result.answer_type),
        evidence=evidence,
        personalization_enabled=personalization_enabled,
    )
    return response


def _handle_local_plan_action(
    *,
    request: ChatRequest,
    active_plan: ActiveTripPlan | None,
    mutation: PlanMutation,
    current_data_client: SupportsCurrentData | None,
    chat_store: ChatStore | None,
    assistant_message_id: str,
    context: ConversationContext,
    resolved_turn: ResolvedTurn,
    session_memory: dict,
) -> ChatResponse | None:
    if active_plan is not None and mutation.operation is PlanOperation.REPLAN_DAY:
        if mutation.target_day is None or not (
            1 <= mutation.target_day <= active_plan.duration_days
        ):
            return _active_plan_response(
                request=request,
                active_plan=active_plan,
                assistant_message_id=assistant_message_id,
                answer=(
                    "Bạn muốn sắp xếp lại ngày nào? "
                    f"Lịch trình hiện có {active_plan.duration_days} ngày."
                ),
                chat_store=chat_store,
                context=context,
                resolved_turn=resolved_turn,
                plan_change=None,
            )
        return None
    if active_plan is None or mutation.operation not in {
        PlanOperation.QUERY_PLAN,
        PlanOperation.REMOVE_SLOT,
        PlanOperation.MOVE_SLOT,
        PlanOperation.UPDATE_TIME,
        PlanOperation.UPDATE_CONSTRAINTS,
    }:
        return None

    updated_plan = active_plan
    plan_change: PlanChange | None = None
    if mutation.operation is not PlanOperation.QUERY_PLAN:
        try:
            days, selected_places, plan_change = apply_plan_mutation(
                active_plan,
                mutation,
                [],
            )
        except ValueError:
            answer = (
                "Tôi chưa xác định được duy nhất điểm cần sửa. "
                "Bạn hãy nêu tên điểm, ngày và buổi/giờ của slot đó."
            )
            return _active_plan_response(
                request=request,
                active_plan=active_plan,
                assistant_message_id=assistant_message_id,
                answer=answer,
                chat_store=chat_store,
                context=context,
                resolved_turn=resolved_turn,
                plan_change=None,
            )
        if mutation.operation is PlanOperation.UPDATE_CONSTRAINTS and any(
            value is not None
            for value in (mutation.adults, mutation.children, mutation.rooms)
        ):
            constraints = active_plan.constraints
            refreshed, _ = enrich_current_data(
                AgentResult(
                    answer="",
                    answer_type="itinerary_planning",
                    evidence=selected_places,
                    itinerary=days,
                    query_plan={
                        "duration_days": active_plan.duration_days,
                        "duration_nights": max(active_plan.duration_days - 1, 1),
                    },
                ),
                current_data_client,
                travel_date=active_plan.start_date,
                include_traffic=False,
                adults=mutation.adults
                or int(constraints.get("adults") or constraints.get("party_size") or 2),
                children=(
                    mutation.children
                    if mutation.children is not None
                    else int(constraints.get("children") or 0)
                ),
                rooms=mutation.rooms or int(constraints.get("rooms") or 1),
                force_hotel_refresh=True,
            )
            days = refreshed.itinerary
            selected_places = refreshed.evidence
        if (
            mutation.operation
            in {
                PlanOperation.REMOVE_SLOT,
                PlanOperation.MOVE_SLOT,
                PlanOperation.UPDATE_TIME,
            }
            and plan_change is not None
        ):
            days = invalidate_adjacent_routes(days, plan_change.changed_slot_ids)
            days, _, route_failures = attach_itinerary_transport(
                days,
                current_data_client,
                start_date=active_plan.start_date,
            )
        else:
            route_failures = 0
        updated_plan = build_active_trip_plan(
            itinerary=days,
            evidence=selected_places,
            city=active_plan.city,
            start_date=active_plan.start_date,
            duration_days=active_plan.duration_days,
            operation=mutation.operation,
            previous=active_plan,
            mutation=mutation,
        )
        if route_failures:
            updated_plan.status = "partial"
        if plan_change is not None:
            plan_change = plan_change.model_copy(
                update={"revision": updated_plan.revision}
            )
        _persist_active_trip_plan(
            chat_store,
            request.session_id,
            updated_plan,
            expected_revision=active_plan.revision,
            user_id=getattr(request, "user_id", None),
        )

    answer = _local_plan_answer(updated_plan, mutation.operation)
    response = _active_plan_response(
        request=request,
        active_plan=updated_plan,
        assistant_message_id=assistant_message_id,
        answer=answer,
        chat_store=chat_store,
        context=context,
        resolved_turn=resolved_turn,
        plan_change=plan_change,
    )
    _save_session_memory(
        chat_store,
        request.session_id,
        existing=session_memory,
        summary=resolved_turn.resolution.summary,
        kb_version=request.kb_version,
        user_id=getattr(request, "user_id", None),
    )
    return response


def _active_plan_response(
    *,
    request: ChatRequest,
    active_plan: ActiveTripPlan,
    assistant_message_id: str,
    answer: str,
    chat_store: ChatStore | None,
    context: ConversationContext,
    resolved_turn: ResolvedTurn,
    plan_change: PlanChange | None,
) -> ChatResponse:
    trace = [
        context.trace_event(),
        resolved_turn.trace_event(),
        {
            "node": "active_trip_plan",
            "status": "completed",
            "plan_id": active_plan.plan_id,
            "revision": active_plan.revision,
            "operation": active_plan.last_operation.value,
        },
    ]
    resolved_context = _resolved_context(context.to_dict(), resolved_turn)
    evidence = [
        EvidenceItem.model_validate(item) for item in active_plan.selected_places
    ]
    response = ChatResponse(
        session_id=request.session_id,
        message_id=assistant_message_id,
        answer=answer,
        intent="itinerary_planning",
        orchestration_mode="plan_revision",
        resolved_context=resolved_context,
        kb_version=request.kb_version,
        evidence=evidence,
        itinerary=active_plan.itinerary,
        trace=trace,
        active_trip_plan=active_plan,
        plan_change=plan_change,
        budget_summary=active_plan.budget_summary,
    )
    _save_message(
        chat_store,
        request,
        assistant_message_id,
        "assistant",
        answer,
        city=active_plan.city,
        metadata={
            "kb_version": request.kb_version,
            "place_ids": [item.place_id for item in evidence],
            "itinerary": active_plan.itinerary,
            "active_trip_plan": active_plan.model_dump(mode="json"),
            "plan_change": plan_change.model_dump(mode="json") if plan_change else None,
            "budget_summary": active_plan.budget_summary.model_dump(mode="json")
            if active_plan.budget_summary
            else None,
            "resolved_context": resolved_context,
            "trace": trace,
        },
    )
    return response


def _local_plan_answer(plan: ActiveTripPlan, operation: PlanOperation) -> str:
    if operation is PlanOperation.QUERY_PLAN:
        return f"Đây là lịch trình đang hoạt động, revision {plan.revision}."
    if operation is PlanOperation.REMOVE_SLOT:
        return (
            f"Đã bỏ điểm bạn chọn và tạo revision {plan.revision}. "
            "Các điểm còn lại được giữ nguyên; những chặng liền kề đã được tính lại."
        )
    if operation is PlanOperation.MOVE_SLOT:
        return (
            f"Đã chuyển điểm bạn chọn và tạo revision {plan.revision}. "
            "Các điểm khác được giữ nguyên; những chặng bị ảnh hưởng đã được tính lại."
        )
    if operation is PlanOperation.UPDATE_TIME:
        return (
            f"Đã đổi giờ của điểm bạn chọn và tạo revision {plan.revision}. "
            "Những chặng có thời điểm khởi hành thay đổi đã được tính lại."
        )
    return (
        f"Đã cập nhật ràng buộc và ngân sách cho revision {plan.revision}. "
        "Các khoản chưa có giá xác minh vẫn được đánh dấu là chưa biết."
    )


def _planning_unavailable_answer(
    active_plan: ActiveTripPlan | None,
    *,
    reason: str | None = None,
) -> str:
    if active_plan is not None:
        return (
            "Mình chưa thể áp dụng lần sắp xếp mới vì chưa ghép được đầy đủ "
            "địa điểm, giờ hoạt động và lưu trú phù hợp. Lịch trình hiện tại "
            f"revision {active_plan.revision} vẫn được giữ nguyên."
        )
    if reason == "no_grounded_candidates_in_city":
        return (
            "Mình chưa nhận được đủ dữ liệu địa điểm đã xác minh để sắp xếp "
            "lịch trình ở lần này. Bạn có thể thử lại sau."
        )
    return (
        "Mình chưa thể tạo lịch trình theo từng ngày từ các địa điểm và điều "
        "kiện đã xác minh. Bạn có thể thử lại hoặc điều chỉnh yêu cầu."
    )


def _planning_outcome(
    trace: dict[str, object],
    *,
    retryable: bool = False,
) -> PlanningOutcome:
    status = str(trace.get("status") or "skipped")
    if status not in {"skipped", "completed", "needs_input", "unavailable"}:
        status = "unavailable"
    return PlanningOutcome(
        status=status,
        reason=str(trace.get("reason") or trace.get("missing") or "") or None,
        retryable=retryable,
        candidate_count=_optional_nonnegative_int(trace.get("candidate_count")),
        itinerary_days=_optional_nonnegative_int(trace.get("itinerary_days")),
    )


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _get_active_trip_plan(
    chat_store: ChatStore | None,
    session_id: str,
    *,
    user_id: str | None,
) -> ActiveTripPlan | None:
    if chat_store is None:
        return None
    getter = getattr(chat_store, "get_active_trip_plan", None)
    if getter is None:
        return None
    try:
        return active_plan_from_value(getter(session_id, user_id=user_id))
    except Exception as exc:
        logger.exception(
            "Active trip plan load failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )
        return None


def _persist_active_trip_plan(
    chat_store: ChatStore | None,
    session_id: str,
    plan: ActiveTripPlan,
    *,
    expected_revision: int | None,
    user_id: str | None,
) -> None:
    if chat_store is None:
        return
    writer = getattr(chat_store, "compare_and_set_active_trip_plan", None)
    if writer is None:
        return
    writer(
        session_id,
        plan.model_dump(mode="json"),
        expected_revision=expected_revision,
        user_id=user_id,
    )


def _require_expected_plan_revision(
    session_id: str,
    active_plan: ActiveTripPlan | None,
    expected_revision: int | None,
) -> None:
    if expected_revision is None:
        return
    actual = active_plan.revision if active_plan is not None else None
    if expected_revision != actual:
        raise TripPlanRevisionConflictError(
            session_id=session_id,
            expected_revision=expected_revision,
            actual_revision=actual,
        )


def _is_mutating_plan_operation(operation: PlanOperation) -> bool:
    return operation in {
        PlanOperation.CREATE,
        PlanOperation.REPLACE_SLOT,
        PlanOperation.REMOVE_SLOT,
        PlanOperation.ADD_SLOT,
        PlanOperation.MOVE_SLOT,
        PlanOperation.UPDATE_TIME,
        PlanOperation.UPDATE_CONSTRAINTS,
        PlanOperation.REPLAN_DAY,
        PlanOperation.REPLAN_ALL,
    }


def _plan_discovery_context(
    plan: ActiveTripPlan,
    mutation: PlanMutation,
) -> tuple[dict | None, dict | None]:
    target = locate_target_slot(plan.itinerary, mutation)
    target_slot = target[2] if target is not None else None
    anchor_name = mutation.anchor_place_name or mutation.target_place_name
    anchor = next(
        (
            item
            for item in plan.selected_places
            if (
                target_slot is not None
                and item.get("place_id") == target_slot.get("place_id")
            )
            or (
                anchor_name
                and str(item.get("name") or "").casefold() == anchor_name.casefold()
            )
        ),
        None,
    )
    return target_slot, anchor


def _plan_discovery_message(
    message: str,
    *,
    city: str,
    entity_type: str | None,
    anchor_name: object,
) -> str:
    kind = entity_type or "địa điểm"
    anchor = f" gần {anchor_name}" if anchor_name else ""
    return (
        f"Gợi ý các {kind} khác tại {city}{anchor}, phù hợp với yêu cầu: {message}. "
        "Chỉ trả về các địa điểm ứng viên, không tạo lịch trình."
    )


def _discovery_entity_type(
    mutation: PlanMutation,
    target: dict | None,
) -> str | None:
    if mutation.operation is PlanOperation.ADD_SLOT:
        if mutation.entity_type:
            return mutation.entity_type
        return {
            "meal": "restaurant",
            "cafe_break": "cafe",
            "check_in": "hotel",
            "check_out": "hotel",
        }.get(str(mutation.target_role or ""))
    if target is not None:
        return str(target.get("entity_type") or "") or mutation.entity_type
    return mutation.entity_type


def _load_spatial_candidates(
    kb_client: KbClient,
    *,
    active_plan: ActiveTripPlan | None,
    mutation: PlanMutation,
    target: dict | None,
    anchor: dict | None,
) -> list[dict] | None:
    if (
        active_plan is None
        or anchor is None
        or mutation.operation
        not in {
            PlanOperation.ADD_SLOT,
            PlanOperation.SUGGEST_NEARBY,
            PlanOperation.REPLACE_SLOT,
        }
    ):
        return None
    loader = getattr(kb_client, "nearby", None)
    if loader is None:
        return None
    anchor_id = str(anchor.get("place_id") or "")
    if not anchor_id:
        return None
    entity_type = _discovery_entity_type(mutation, target)
    excluded = [
        str(slot.get("place_id"))
        for day_item in active_plan.itinerary
        for slot in day_item.get("slots", [])
        if slot.get("place_id")
    ]
    try:
        payload = loader(
            anchor_place_id=anchor_id,
            entity_types=[entity_type] if entity_type else None,
            city=active_plan.city,
            radius_km=5.0,
            excluded_place_ids=excluded,
            limit=12,
            kb_version="v8",
        )
    except Exception as exc:
        logger.warning(
            "Nearby lookup degraded anchor={} error_type={}",
            anchor_id,
            exc.__class__.__name__,
        )
        return None
    return [
        _nearby_evidence_row(item)
        for item in payload.get("items", [])
        if isinstance(item, dict) and item.get("place_id")
    ]


def _nearby_evidence_row(item: dict) -> dict:
    coordinates = item.get("coordinates")
    prices = item.get("prices")
    safe_attributes = item.get("attributes")
    attributes = dict(safe_attributes) if isinstance(safe_attributes, dict) else {}
    if isinstance(coordinates, dict):
        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")
        attributes.update(
            {
                "lat": latitude,
                "latitude": latitude,
                "lng": longitude,
                "longitude": longitude,
            }
        )
    if isinstance(prices, dict):
        attributes.update(
            {
                key: value
                for key, value in prices.items()
                if value is not None and key not in {"currency", "display_text"}
            }
        )
        if prices.get("display_text"):
            attributes["price_range"] = prices["display_text"]
    return {
        "place_id": item.get("place_id"),
        "name": item.get("name"),
        "city": item.get("city"),
        "entity_type": item.get("entity_type"),
        "category": item.get("category"),
        "distance_km": item.get("distance_km"),
        "source": item.get("source") or {},
        "attributes": attributes,
    }


def _merge_evidence_rows(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for row in (item for group in groups for item in group):
        place_id = str(row.get("place_id") or "")
        if not place_id or place_id in seen:
            continue
        seen.add(place_id)
        merged.append(row)
    return merged


def _itinerary_city(itinerary: list[dict]) -> str:
    return next(
        (
            str(slot.get("city"))
            for day_item in itinerary
            for slot in day_item.get("slots", [])
            if slot.get("city")
        ),
        "Không xác định",
    )


def _itinerary_duration(itinerary: list[dict]) -> int:
    return max((int(day.get("day") or 1) for day in itinerary), default=1)


def _merge_replanned_day(
    plan: ActiveTripPlan,
    generated: list[dict],
    *,
    target_day: int,
) -> list[dict]:
    if not 1 <= target_day <= plan.duration_days:
        raise ValueError("plan_replan_day_out_of_range")
    replacement = next(
        (
            day_item
            for day_item in generated
            if int(day_item.get("day") or 0) == target_day
        ),
        generated[0] if len(generated) == 1 else None,
    )
    if replacement is None:
        raise ValueError("plan_replan_day_missing")
    revised_day = deepcopy(replacement)
    revised_day["day"] = target_day
    days = [
        deepcopy(day_item)
        for day_item in plan.itinerary
        if int(day_item.get("day") or 0) != target_day
    ]
    days.append(revised_day)
    return sorted(days, key=lambda item: int(item.get("day") or 0))


def _slot_ids_for_day(itinerary: list[dict], day: int) -> list[str]:
    return [
        str(slot.get("slot_id"))
        for day_item in itinerary
        if int(day_item.get("day") or 0) == day
        for slot in day_item.get("slots", [])
        if slot.get("slot_id")
    ]


def _all_slot_ids(plan: ActiveTripPlan) -> list[str]:
    return [
        str(slot.get("slot_id"))
        for day_item in plan.itinerary
        for slot in day_item.get("slots", [])
        if slot.get("slot_id")
    ]


def _itinerary_place_signature(itinerary: list[dict]) -> tuple:
    return tuple(
        (
            int(day_item.get("day") or 0),
            tuple(
                (
                    int(slot.get("order") or index),
                    str(slot.get("place_id") or ""),
                )
                for index, slot in enumerate(day_item.get("slots", []), start=1)
            ),
        )
        for day_item in sorted(
            itinerary,
            key=lambda item: int(item.get("day") or 0),
        )
    )


def _resolved_context(context: dict, resolved_turn) -> dict:
    return context | {
        "conversation_route": resolved_turn.resolution.route,
        "contextualized": (
            resolved_turn.resolution.standalone_message.strip()
            != resolved_turn.original_message.strip()
        ),
    }


def _referenced_entities(evidence: list[EvidenceItem]) -> list[dict[str, str]]:
    return [
        {
            "id": item.place_id,
            "name": str(item.name),
            "kind": item.entity_type or item.category or "place",
        }
        for item in evidence
        if item.name
    ][:20]


def _save_message(
    chat_store: ChatStore | None,
    request: ChatRequest,
    message_id: str,
    role: str,
    content: str,
    city: str | None,
    metadata: dict | None = None,
) -> None:
    if chat_store is None:
        return
    try:
        chat_store.save_message(
            request.session_id,
            message_id,
            role,
            content,
            user_id=getattr(request, "user_id", None),
            city=city,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception(
            "Chat persistence failed session_id={} role={} error_type={}",
            request.session_id,
            role,
            exc.__class__.__name__,
        )


def _recent_messages(
    chat_store: ChatStore | None,
    session_id: str,
    limit: int,
    *,
    user_id: str | None = None,
) -> list[dict]:
    if chat_store is None:
        return []
    try:
        return chat_store.recent_messages(session_id, limit, user_id=user_id)
    except Exception as exc:
        logger.exception(
            "Chat history load failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )
        return []


def _get_session_memory(
    chat_store: ChatStore | None,
    session_id: str,
    *,
    user_id: str | None = None,
) -> dict:
    if chat_store is None:
        return {}
    try:
        return chat_store.get_session_memory(session_id, user_id=user_id)
    except Exception as exc:
        logger.exception(
            "Conversation memory load failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )
        return {}


def _kb_conversation_context(
    *,
    session_memory: dict,
    context: ConversationContext,
    resolved_turn: ResolvedTurn,
    personalization: dict | None = None,
) -> dict | None:
    stored = session_memory.get("kb_context")
    # A previous query's intent, duration, or entity filters must never lock a
    # new turn into the old plan. Durable preferences live in personalization;
    # entity references are resolved into the standalone query by Gemini.
    payload = {
        key: value
        for key, value in (stored.items() if isinstance(stored, dict) else [])
        if key in {"turn_count", "cities", "city_source", "personalization"}
    }
    if context.city:
        payload["cities"] = [context.city]
        payload["city_source"] = context.city_source
    if context.history_messages and not payload.get("turn_count"):
        payload["turn_count"] = context.history_messages
    standalone = resolved_turn.resolution.standalone_message.strip()
    if standalone and standalone != resolved_turn.original_message.strip():
        payload["resolved_query"] = standalone
    if personalization:
        payload["personalization"] = personalization
    else:
        payload.pop("personalization", None)
    return payload or None


def _save_session_memory(
    chat_store: ChatStore | None,
    session_id: str,
    *,
    existing: dict,
    summary: str | None = None,
    kb_context: dict | None = None,
    kb_version: str | None = None,
    user_id: str | None = None,
) -> None:
    if chat_store is None or (not summary and not kb_context and not kb_version):
        return
    memory = dict(existing)
    if summary:
        memory["summary"] = summary
    if kb_context:
        memory["kb_context"] = kb_context
    if kb_version:
        memory["kb_version"] = kb_version
    try:
        chat_store.save_session_memory(
            session_id,
            memory,
            user_id=user_id,
        )
    except Exception as exc:
        logger.exception(
            "Conversation memory save failed session_id={} error_type={}",
            session_id,
            exc.__class__.__name__,
        )


def _record_grounded_place_interest(
    store: UserProfileStore | None,
    *,
    user_id: str | None,
    session_id: str,
    answer_type: str,
    evidence: list[EvidenceItem],
    personalization_enabled: bool,
) -> None:
    if (
        store is None
        or not user_id
        or not personalization_enabled
        or answer_type != "entity_detail"
    ):
        return
    for item in evidence[:3]:
        try:
            store.save_event(
                user_id,
                PreferenceEvent(
                    event_id=uuid4().hex,
                    event_type="ask_place",
                    place_id=item.place_id,
                    session_id=session_id,
                ),
            )
        except Exception as exc:
            logger.warning(
                "Preference event save failed user_id={} place_id={} error_type={}",
                user_id,
                item.place_id,
                exc.__class__.__name__,
            )
