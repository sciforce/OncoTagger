"""Reproduce the bounded sensitivity analysis for 300 task-unassigned records.

The public input contains only arbitrary validation IDs and derived labels/flags. It
does not contain Web of Science titles, abstracts, authors, addresses, or DOIs.
The baseline corpus summaries are aggregate inputs; this script applies the manual
consensus labels to that single unassigned stratum and verifies the published output.
"""

from __future__ import annotations

import argparse
import csv
import math
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PUBLISHED_PATH = ROOT / "aggregate" / "task_unassigned_sensitivity.csv"
REDACTED_PATH = ROOT / "redacted_labels" / "task_unassigned_300_redacted.csv"

BASELINE = "baseline_pipeline"
CORRECTED = "manual_correction_of_300_unassigned_only"
TASKS = [
    "classification",
    "segmentation",
    "prognosis",
    "synthesis",
    "genomic",
    "integration",
    "nlp",
    "auxiliary",
    "unassigned",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["table"]),
        str(row.get("publication_year", "")),
        str(row["task"]),
    )


def integer(value: object) -> int:
    return int(float(str(value)))


def reproduce() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    published = read_rows(PUBLISHED_PATH)
    baseline_rows = [row for row in published if row["scenario"] == BASELINE]
    expected_rows = [row for row in published if row["scenario"] == CORRECTED]
    redacted = read_rows(REDACTED_PATH)

    if len(redacted) != 300:
        raise AssertionError(f"Expected 300 redacted records, found {len(redacted)}")
    if len({row["validation_id"] for row in redacted}) != 300:
        raise AssertionError("Redacted validation IDs are not unique")

    calculated: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in baseline_rows:
        copied: dict[str, object] = deepcopy(row)
        copied["scenario"] = CORRECTED
        copied["count"] = integer(row["count"])
        copied["denominator"] = integer(row["denominator"])
        calculated[key(copied)] = copied

    def adjust(
        table: str,
        year: str,
        task: str,
        *,
        count_delta: int = 0,
        denominator_delta: int = 0,
    ) -> None:
        row = calculated[(table, year, task)]
        row["count"] = int(row["count"]) + count_delta
        row["denominator"] = int(row["denominator"]) + denominator_delta

    for record in redacted:
        target = record["manual_primary_task"]
        if target not in TASKS:
            raise AssertionError(f"Unknown task label: {target}")
        if target == "unassigned":
            continue

        year = record["publication_year"]
        no_metric = integer(record["no_metrics_reported"])
        candidate = integer(record["candidate_translational_signal"])

        adjust("overall_task_distribution", "", "unassigned", count_delta=-1)
        adjust("overall_task_distribution", "", target, count_delta=1)
        adjust("task_by_year", year, "unassigned", count_delta=-1)
        adjust("task_by_year", year, target, count_delta=1)

        adjust(
            "no_metric_rate_by_task",
            "",
            "unassigned",
            count_delta=-no_metric,
            denominator_delta=-1,
        )
        adjust(
            "no_metric_rate_by_task",
            "",
            target,
            count_delta=no_metric,
            denominator_delta=1,
        )

        if candidate:
            adjust(
                "candidate_translational_subset_by_task",
                "",
                "unassigned",
                count_delta=-1,
            )
            adjust(
                "candidate_translational_subset_by_task",
                "",
                target,
                count_delta=1,
            )

    ordered = []
    for baseline in baseline_rows:
        row = calculated[key(baseline)]
        denominator = int(row["denominator"])
        row["value"] = int(row["count"]) / denominator if denominator else ""
        ordered.append(row)

    expected = {key(row): row for row in expected_rows}
    if set(expected) != set(calculated):
        raise AssertionError("Published and reproduced sensitivity rows differ")
    for row in ordered:
        published_row = expected[key(row)]
        if int(row["count"]) != integer(published_row["count"]):
            raise AssertionError(f"Count mismatch for {key(row)}")
        if int(row["denominator"]) != integer(published_row["denominator"]):
            raise AssertionError(f"Denominator mismatch for {key(row)}")
        if not math.isclose(
            float(row["value"]),
            float(published_row["value"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Value mismatch for {key(row)}")
    return ordered, baseline_rows


def write_output(
    path: Path,
    baseline_rows: list[dict[str, str]],
    corrected_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "table",
        "scenario",
        "publication_year",
        "task",
        "metric_source",
        "count",
        "denominator",
        "value",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(baseline_rows)
        writer.writerows(corrected_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a reproduced baseline-plus-corrected CSV.",
    )
    args = parser.parse_args()
    corrected, baseline = reproduce()
    if args.output:
        write_output(args.output, baseline, corrected)
    print(
        "PASS: reproduced all bounded task-unassigned sensitivity rows from "
        "300 redacted consensus labels and aggregate baseline inputs."
    )


if __name__ == "__main__":
    main()
