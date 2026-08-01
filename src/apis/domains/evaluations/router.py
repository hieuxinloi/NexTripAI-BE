from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from src.apis.domains.chat.router import select_kb_version
from src.apis.domains.evaluations.manager import (
    EvaluationAlreadyRunningError,
    EvaluationManager,
)
from src.apis.domains.evaluations.schemas import (
    EvaluationHistoryResponse,
    EvaluationJobResponse,
)
from src.apis.domains.evaluations.workbook import (
    MAX_WORKBOOK_BYTES,
    WorkbookValidationError,
    parse_evaluation_workbook,
)
from src.security.auth import Principal, require_admin


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.get("", response_model=EvaluationHistoryResponse)
async def evaluation_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    principal: Principal = Depends(require_admin),
) -> EvaluationHistoryResponse:
    manager: EvaluationManager = request.app.state.evaluation_manager
    evaluations = await manager.list_history(
        owner_id=principal.uid,
        limit=limit,
    )
    return EvaluationHistoryResponse(evaluations=evaluations)


@router.post(
    "",
    response_model=EvaluationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_evaluation(
    request: Request,
    file: UploadFile = File(...),
    kb_version: str = Form(...),
    principal: Principal = Depends(require_admin),
) -> EvaluationJobResponse:
    content = await file.read(MAX_WORKBOOK_BYTES + 1)
    try:
        workbook = parse_evaluation_workbook(file.filename or "", content)
    except WorkbookValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    selected_version = select_kb_version(
        kb_version,
        request,
        client_selected=True,
        principal=principal,
    )
    manager: EvaluationManager = request.app.state.evaluation_manager
    try:
        return manager.start(
            workbook=workbook,
            filename=file.filename or "evaluation.xlsx",
            kb_version=selected_version,
            owner_id=principal.uid,
        )
    except EvaluationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{job_id}", response_model=EvaluationJobResponse)
async def evaluation_status(
    job_id: str,
    request: Request,
    principal: Principal = Depends(require_admin),
) -> EvaluationJobResponse:
    manager: EvaluationManager = request.app.state.evaluation_manager
    try:
        return await manager.get(job_id, owner_id=principal.uid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt đánh giá.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Bạn không có quyền xem lượt đánh giá này.") from exc


@router.delete("/{job_id}", response_model=EvaluationJobResponse)
async def cancel_evaluation(
    job_id: str,
    request: Request,
    principal: Principal = Depends(require_admin),
) -> EvaluationJobResponse:
    manager: EvaluationManager = request.app.state.evaluation_manager
    try:
        return await manager.cancel(job_id, owner_id=principal.uid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt đánh giá.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Bạn không có quyền hủy lượt đánh giá này.") from exc
