from __future__ import annotations

from io import BytesIO

import pytest
from openpyxl import Workbook

from src.apis.domains.evaluations.workbook import (
    WorkbookValidationError,
    parse_evaluation_workbook,
)


def _workbook_bytes(rows: list[tuple[object, object]]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Test cases"
    sheet.append(["Câu hỏi của người dùng", "Kết quả mong đợi"])
    for row in rows:
        sheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def test_parser_reads_expected_evaluation_columns() -> None:
    parsed = parse_evaluation_workbook(
        "cases.xlsx",
        _workbook_bytes(
            [
                ("Quy Nhơn có gì?", "Nêu hai địa điểm."),
                ("Ăn gì ở Đà Nẵng?", "Gợi ý món địa phương."),
            ]
        ),
    )

    assert parsed.sheet_name == "Test cases"
    assert [item.row_number for item in parsed.cases] == [2, 3]
    assert parsed.cases[0].question == "Quy Nhơn có gì?"


def test_parser_rejects_partial_rows() -> None:
    with pytest.raises(WorkbookValidationError, match="Dòng 2"):
        parse_evaluation_workbook(
            "cases.xlsx",
            _workbook_bytes([("Quy Nhơn có gì?", None)]),
        )


def test_parser_rejects_wrong_headers() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Question", "Expected"])
    sheet.append(["Question 1", "Expected 1"])
    output = BytesIO()
    workbook.save(output)
    workbook.close()

    with pytest.raises(WorkbookValidationError, match="Không tìm thấy sheet"):
        parse_evaluation_workbook("cases.xlsx", output.getvalue())
