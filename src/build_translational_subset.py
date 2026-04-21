from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "results" / "filtered_dataset_binary_classification.xlsx"
OUTPUT_XLSX = PROJECT_ROOT / "documentation" / "article" / "supplementary_translational_subset.xlsx"
OUTPUT_JSON = PROJECT_ROOT / "documentation" / "self-documented docs" / "translational_subset_summary.json"
OUTPUT_TXT = PROJECT_ROOT / "documentation" / "self-documented docs" / "translational_subset_summary.txt"


TITLE_PATTERNS: dict[str, str] = {
    "prospective_title_signal": r"\bprospective\b",
    "real_world_title_signal": r"\breal[- ]world\b",
    "external_validation_title_signal": r"\bexternal validation|externally validated|validation study\b",
    "interface_title_signal": r"\bweb[- ]based|online|calculator|app|shiny|streamlit|decision support system|decision-support|decision support|platform|tool|interface\b",
    "multicenter_title_signal": r"\b(?:multicenter|multi-center|multicentre|multi-centre|dual-center|dual-centre|bicenter|two-center|two-centre|national)\b",
}


def load_workbook() -> pd.DataFrame:
    header = pd.read_excel(INPUT_PATH, nrows=0)
    metric_context_cols = [c for c in header.columns if c.startswith("metric_context_")]
    wanted = {
        "Article Title",
        "Abstract",
        "Source Title",
        "Publication Year",
        "DOI",
        "primary_task",
        "weighted_category",
        "composite_source",
    } | set(metric_context_cols)
    return pd.read_excel(INPUT_PATH, usecols=lambda c: c in wanted)


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    title = df["Article Title"].fillna("").astype(str)
    for name, pattern in TITLE_PATTERNS.items():
        df[name] = title.str.contains(pattern, case=False, na=False, regex=True)

    metric_context_cols = [c for c in df.columns if c.startswith("metric_context_")]
    ctx_frame = df[metric_context_cols].fillna("").astype(str)
    df["external_validation_context_signal"] = ctx_frame.apply(
        lambda s: s.str.contains("external_validation", case=False, na=False)
    ).any(axis=1)

    df["high_confidence_translational_signal"] = (
        df["external_validation_context_signal"]
        & (
            df["prospective_title_signal"]
            | df["real_world_title_signal"]
            | df["multicenter_title_signal"]
            | df["interface_title_signal"]
            | df["external_validation_title_signal"]
        )
    )

    df["signal_score"] = (
        df["external_validation_context_signal"].astype(int) * 3
        + df["prospective_title_signal"].astype(int) * 3
        + df["real_world_title_signal"].astype(int) * 3
        + df["multicenter_title_signal"].astype(int) * 2
        + df["interface_title_signal"].astype(int) * 2
        + df["external_validation_title_signal"].astype(int) * 2
        + df["weighted_category"]
        .astype(str)
        .str.lower()
        .map({"very high": 2, "high": 1, "medium": 0, "low": 0, "very low": 0})
        .fillna(0)
        .astype(int)
    )

    return df


def build_summary(df: pd.DataFrame) -> dict:
    high_conf = df[df["high_confidence_translational_signal"]].copy()
    high_conf_years = high_conf["Publication Year"].value_counts().sort_index().to_dict()
    high_conf_tasks = high_conf["primary_task"].fillna("NaN").value_counts().to_dict()
    high_conf_weighted = high_conf["weighted_category"].fillna("NaN").value_counts().to_dict()

    return {
        "input_rows": int(len(df)),
        "external_validation_context_signal": int(df["external_validation_context_signal"].sum()),
        "prospective_title_signal": int(df["prospective_title_signal"].sum()),
        "real_world_title_signal": int(df["real_world_title_signal"].sum()),
        "multicenter_title_signal": int(df["multicenter_title_signal"].sum()),
        "interface_title_signal": int(df["interface_title_signal"].sum()),
        "external_validation_title_signal": int(df["external_validation_title_signal"].sum()),
        "high_confidence_translational_subset": int(len(high_conf)),
        "high_confidence_high_or_very_high": int(
            high_conf["weighted_category"]
            .astype(str)
            .str.lower()
            .isin(["high", "very high"])
            .sum()
        ),
        "high_confidence_years": high_conf_years,
        "high_confidence_tasks": high_conf_tasks,
        "high_confidence_weighted_category": high_conf_weighted,
        "selection_rule": (
            "high_confidence_translational_signal = external_validation_context_signal AND "
            "(prospective_title_signal OR real_world_title_signal OR multicenter_title_signal "
            "OR interface_title_signal OR external_validation_title_signal)"
        ),
    }


def write_outputs(df: pd.DataFrame, summary: dict) -> None:
    selected_cols = [
        "Publication Year",
        "primary_task",
        "weighted_category",
        "composite_source",
        "Source Title",
        "DOI",
        "Article Title",
        "prospective_title_signal",
        "real_world_title_signal",
        "external_validation_title_signal",
        "interface_title_signal",
        "multicenter_title_signal",
        "external_validation_context_signal",
        "high_confidence_translational_signal",
        "signal_score",
    ]

    external_validation = (
        df[df["external_validation_context_signal"]]
        .copy()
        .sort_values(["signal_score", "Publication Year"], ascending=[False, False])
    )
    high_conf = (
        df[df["high_confidence_translational_signal"]]
        .copy()
        .sort_values(["signal_score", "Publication Year"], ascending=[False, False])
    )
    top_examples = high_conf.head(50).copy()

    summary_rows = [
        {"Metric": key, "Value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
    ]
    summary_df = pd.DataFrame(summary_rows)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        external_validation[selected_cols].to_excel(writer, sheet_name="ExternalValidation755", index=False)
        high_conf[selected_cols].to_excel(writer, sheet_name="HighConfidence225", index=False)
        top_examples[selected_cols].to_excel(writer, sheet_name="TopExamples50", index=False)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    text_lines = [
        "Supplementary Translational Subset Summary",
        "=========================================",
        "",
        f"Input rows: {summary['input_rows']}",
        f"Rows with external_validation context signal: {summary['external_validation_context_signal']}",
        f"Prospective title signal: {summary['prospective_title_signal']}",
        f"Real-world title signal: {summary['real_world_title_signal']}",
        f"Multicenter title signal: {summary['multicenter_title_signal']}",
        f"Interface title signal: {summary['interface_title_signal']}",
        f"External-validation title signal: {summary['external_validation_title_signal']}",
        f"High-confidence translational subset: {summary['high_confidence_translational_subset']}",
        f"High-confidence subset with high/very high weighted category: {summary['high_confidence_high_or_very_high']}",
        "",
        "Selection rule:",
        summary["selection_rule"],
        "",
        "High-confidence subset by year:",
    ]
    for year, count in summary["high_confidence_years"].items():
        text_lines.append(f"- {year}: {count}")
    text_lines.extend(["", "High-confidence subset by primary task:"])
    for task, count in summary["high_confidence_tasks"].items():
        text_lines.append(f"- {task}: {count}")
    text_lines.extend(["", "High-confidence subset by weighted category:"])
    for category, count in summary["high_confidence_weighted_category"].items():
        text_lines.append(f"- {category}: {count}")

    OUTPUT_TXT.write_text("\n".join(text_lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_workbook()
    df = add_flags(df)
    summary = build_summary(df)
    write_outputs(df, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {OUTPUT_XLSX}")
    print(f"[OK] Wrote {OUTPUT_JSON}")
    print(f"[OK] Wrote {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
