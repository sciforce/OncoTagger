from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).with_name("outputs")
DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"

ANALYSIS_XLSX = ROOT / "data" / "results" / "filtered_dataset_binary_classification_analysis.xlsx"
POPULATION_XLSX = ROOT / "data" / "results" / "article to population ratio.xlsx"
SUMMARY_JSON = ROOT / "data" / "supplementary material" / "current_full_pipeline_summary.json"
TRANSLATIONAL_JSON = ROOT / "data" / "supplementary material" / "translational_subset_summary.json"
PRIMARY_VALIDATION_JSON = ROOT / "data" / "manual validation" / "primary_validation_400_analysis.json"
DETECTION_AUDIT_JSON = ROOT / "data" / "manual validation" / "secondary_detection_audit_analysis.json"


COLORS = {
    "blue": "#0b63b6",
    "orange": "#ff6b00",
    "green": "#008f83",
    "purple": "#8e24aa",
    "pink": "#f781bf",
    "brown": "#a65628",
    "gray": "#707070",
    "light_gray": "#f2f2f2",
    "dark": "#222222",
}

PALETTE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["pink"],
    COLORS["brown"],
    "#66c2a5",
    "#e6ab02",
]

DESCRIPTOR_LABEL_OVERRIDES = {
    "SVM": "Support vector machines",
    "U Net Family": "U-Net Family",
    "roc-auc": "ROC-AUC",
    "accuracy": "accuracy",
    "sensitivity": "sensitivity",
    "dice": "Dice",
    "f1-score": "F1-score",
    "precision": "precision",
    "c-index": "C-index",
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        return plt, FancyArrowPatch, FancyBboxPatch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = DATA_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_sheet(sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(ANALYSIS_XLSX, sheet_name=sheet_name)


def clean_label(value: object) -> str:
    text = str(value).replace("_", " ").replace("NaN", "not available")
    text = text.replace("Cancer", "cancer")
    return text


def descriptor_label(value: object) -> str:
    label = clean_label(value)
    return DESCRIPTOR_LABEL_OVERRIDES.get(label, label)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def write_display_plan() -> None:
    rows = [
        {
            "display_item": "Figure 1",
            "placement": "main",
            "title": "PRISMA-inspired corpus flow",
            "source": "current_full_pipeline_summary.json; info for PRISMA schema.txt",
            "rationale": "Shows the screening denominator, automated/manual split, and final analytic corpus.",
        },
        {
            "display_item": "Figure 2",
            "placement": "main",
            "title": "Temporal growth task redistribution and metric reporting",
            "source": "Tasks by Year; No Metrics by Year",
            "rationale": "Captures the main temporal story without dense cross-tabs.",
        },
        {
            "display_item": "Figure 3",
            "placement": "main",
            "title": "Top corpus descriptors dashboard",
            "source": "table2_top_corpus_descriptors.csv derived from the analysis workbook",
            "rationale": "Replaces a dense descriptor table with a reproducible four-panel figure.",
        },
        {
            "display_item": "Figure 4",
            "placement": "main",
            "title": "Geography and translational signals",
            "source": "Reprint Country Overall; article to population ratio.xlsx; translational_subset_summary.json",
            "rationale": "Separates raw publication geography, per-capita output, and implementation-oriented signals.",
        },
        {
            "display_item": "Table 1",
            "placement": "main",
            "title": "Study design denominators and validation summary",
            "source": "summary JSON; validation JSON",
            "rationale": "Gives reviewers one compact denominator and validation table.",
        },
        {
            "display_item": "Supplementary Table 1",
            "placement": "supplementary",
            "title": "Top corpus descriptors source table",
            "source": "analysis workbook",
            "rationale": "Source table for the Figure 3 descriptor dashboard.",
        },
        {
            "display_item": "Supplementary Table 2",
            "placement": "supplementary",
            "title": "Full country rankings",
            "source": "Reprint Country Overall; article to population ratio.xlsx",
            "rationale": "Too dense for main text.",
        },
        {
            "display_item": "Supplementary Table 3",
            "placement": "supplementary",
            "title": "Full cancer task and AI cross-tabs",
            "source": "Task x Cancer; Task x AI Models; Task x AI Classes",
            "rationale": "Useful for reuse but too wide for the Article display limit.",
        },
        {
            "display_item": "Supplementary Table 4",
            "placement": "supplementary",
            "title": "Metric ecology and validation confusion matrices",
            "source": "Composite Src sheets; manual-validation workbooks",
            "rationale": "Supports methods credibility without crowding the main results.",
        },
        {
            "display_item": "Supplementary Table 5",
            "placement": "supplementary",
            "title": "Translational subset article details",
            "source": "supplementary_translational_subset.xlsx",
            "rationale": "Article-level examples are interpretive support, not the core corpus map.",
        },
    ]
    save_csv(pd.DataFrame(rows), "main_display_plan.csv")


def build_table_data() -> None:
    summary = load_json(SUMMARY_JSON)["summary"]
    full_corpus_n = int(summary["filtered_rows_input_to_main"])
    primary = load_json(PRIMARY_VALIDATION_JSON)
    detection = load_json(DETECTION_AUDIT_JSON)

    weighted = primary["manual_vs_weighted_category"]
    composite = primary["manual_vs_composite_metric"]

    table1 = pd.DataFrame(
        [
            ("Records identified from Web of Science exports", "59,994", "Raw merged WoS records"),
            ("Duplicate records removed", "38", "37 DOI duplicates and 1 title/year fallback duplicate"),
            ("Publication-year 2026 records excluded", "128", "Out-of-scope future-year records"),
            ("Records screened after deduplication and year filtering", "59,828", "Screening denominator"),
            ("Records included by automated filtering", "20,723", "Before manual-review additions"),
            ("Manual-review records retained", "43 of 48", "Borderline records adjudicated by authors"),
            ("Final filtered corpus", f"{summary['filtered_rows_input_to_main']:,}", "Main descriptive denominator"),
            ("Abstracts with at least one detected metric", f"{summary['no_metrics_reported_0_metric_bearing']:,}", "60.4% of final corpus"),
            ("Scoreable performance-category corpus", f"{summary['weighted_category_available']:,}", "Weighted/composite category available"),
            ("Primary ordinal validation sample", f"{primary['N']}", "Manual category labels"),
            ("Manual versus composite exact agreement", pct(composite["exact_agreement"]), "N=379 comparable records"),
            ("Manual versus composite linear weighted kappa", f"{composite['linear_weighted_kappa']:.3f}", "Ordinal agreement"),
            ("Metric-detection audit sample", f"{detection['N_total_audited']}", "Stratified 200-record audit"),
            ("Metric-detection accuracy", pct(detection["accuracy"]), "Sensitivity 84.6%; specificity 98.8%"),
        ],
        columns=["Item", "Value", "Note"],
    )
    save_csv(table1, "table1_design_validation_summary.csv")

    task = read_sheet("Task Categories Frequency").head(8)
    cancer = read_sheet("Cancer Types Frequency").head(10)
    ai = read_sheet("AI Models Frequency").head(10)
    metric = read_sheet("Composite Src Overall").head(8)

    def descriptor_rows(df: pd.DataFrame, group: str) -> list[dict]:
        label_col = df.columns[0]
        count_col = "Count" if "Count" in df.columns else df.columns[1]
        rows = []
        for _, row in df.iterrows():
            count = int(row[count_col])
            rows.append(
                {
                    "Descriptor group": group,
                    "Descriptor": clean_label(row[label_col]),
                    "Count": count,
                    "Share of full corpus (%)": round(count / full_corpus_n * 100, 2),
                }
            )
        return rows

    table2 = pd.DataFrame(
        descriptor_rows(task, "Primary task")
        + descriptor_rows(cancer, "Cancer site")
        + descriptor_rows(ai, "AI family")
        + descriptor_rows(metric, "Composite metric source")
    )
    save_csv(table2, "table2_top_corpus_descriptors.csv")


def build_flow_figure() -> None:
    plt, FancyArrowPatch, FancyBboxPatch = require_matplotlib()

    flow = pd.DataFrame(
        [
            ("identified", "Records identified from WoS exports", 59994),
            ("duplicates_removed", "Duplicate records removed", 38),
            ("year_2026_removed", "Publication-year 2026 records removed", 128),
            ("screened", "Records screened after deduplication and year filtering", 59828),
            ("auto_included", "Included by automated bibliographic filtering", 20723),
            ("manual_review", "Assigned to manual review", 48),
            ("auto_excluded", "Excluded by automated bibliographic filtering", 39057),
            ("manual_retained", "Manual-review records retained", 43),
            ("manual_not_included", "Manual-review records not included", 5),
            ("final_corpus", "Final filtered corpus", 20766),
            ("metric_bearing", "Abstracts with at least one detected metric", 12538),
            ("scoreable", "Scoreable performance-category corpus", 12225),
            ("no_metrics", "No abstract-level metric detected", 8228),
            ("metric_no_category", "Detected metric but no final category", 313),
        ],
        columns=["step_id", "label", "count"],
    )
    save_csv(flow, "figure1_prisma_flow.csv")

    fig, ax = plt.subplots(figsize=(8.0, 11.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.set_axis_off()

    main_fc = "#eaf2fb"
    exclusion_fc = "#f5f5f5"
    manual_fc = "#fff0dd"
    phase_fc = "#eef2f6"

    def box(
        cx: float,
        cy: float,
        w: float,
        h: float,
        title: str,
        count: int,
        fc: str,
        ec: str = COLORS["dark"],
        wrap_width: int = 30,
    ) -> None:
        x = cx - w / 2
        y = cy - h / 2
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(cx, cy + h * 0.13, wrap(title, wrap_width), ha="center", va="center", fontsize=9.4)
        ax.text(cx, cy - h * 0.30, f"n = {count:,}", ha="center", va="center", fontsize=10.5, fontweight="bold")

    def phase_label(label: str, y0: float, y1: float) -> None:
        patch = FancyBboxPatch(
            (0.25, y0),
            0.48,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.8,
            edgecolor="#c9d2df",
            facecolor=phase_fc,
        )
        ax.add_patch(patch)
        ax.text(
            0.49,
            (y0 + y1) / 2,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.7,
            fontweight="bold",
            color="#44556a",
        )

    def connector(points: list[tuple[float, float]], color: str = COLORS["dark"]) -> None:
        if len(points) < 2:
            return
        if len(points) > 2:
            xs = [point[0] for point in points[:-1]]
            ys = [point[1] for point in points[:-1]]
            ax.plot(xs, ys, color=color, linewidth=1.15, solid_capstyle="round", zorder=0)
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.15,
                color=color,
                shrinkA=0,
                shrinkB=3,
                zorder=0,
            )
        )

    phase_label("Identification", 13.85, 15.25)
    phase_label("Screening", 10.15, 13.05)
    phase_label("Manual review", 7.85, 9.95)
    phase_label("Final corpus", 5.75, 6.95)
    phase_label("Metric analysis", 1.85, 4.55)

    box(4.15, 14.65, 4.35, 1.05, "Records identified from Web of Science exports", 59994, main_fc)
    box(8.20, 14.95, 2.65, 0.78, "Duplicate records removed", 38, exclusion_fc, wrap_width=22)
    box(8.20, 13.90, 2.65, 0.85, "Publication-year 2026 records excluded", 128, exclusion_fc, wrap_width=22)
    box(4.15, 12.55, 4.35, 1.05, "Records screened after deduplication and year filtering", 59828, main_fc)

    box(2.15, 10.65, 2.75, 1.05, "Included by automated bibliographic filtering", 20723, main_fc, wrap_width=23)
    box(5.00, 10.65, 2.25, 1.05, "Assigned to manual review", 48, manual_fc, wrap_width=20)
    box(8.05, 10.65, 2.85, 1.05, "Excluded by automated bibliographic filtering", 39057, exclusion_fc, wrap_width=23)

    box(3.65, 8.55, 2.50, 1.00, "Manual-review records retained", 43, manual_fc, wrap_width=22)
    box(6.55, 8.55, 2.50, 1.00, "Manual-review records not included", 5, exclusion_fc, wrap_width=22)
    box(4.95, 6.35, 4.55, 1.05, "Final filtered corpus", 20766, main_fc)

    box(3.10, 4.10, 2.85, 1.00, "Abstracts with at least one detected metric", 12538, main_fc, wrap_width=24)
    box(6.90, 4.10, 2.85, 1.00, "No abstract-level metric detected", 8228, exclusion_fc, wrap_width=24)
    box(3.10, 2.35, 2.85, 1.00, "Scoreable performance-category corpus", 12225, main_fc, wrap_width=24)

    ax.text(
        5.0,
        1.25,
        "An additional 313 metric-bearing records had a detected metric but no final weighted or composite category.",
        ha="center",
        va="center",
        fontsize=8.3,
        color=COLORS["gray"],
    )
    ax.text(
        5.0,
        0.65,
        "PRISMA-inspired automated bibliographic screening flow, not a full systematic review.",
        ha="center",
        va="center",
        fontsize=8.3,
        color=COLORS["gray"],
        style="italic",
    )

    connector([(4.15, 14.13), (4.15, 13.08)])
    connector([(6.33, 14.85), (6.75, 14.85), (6.75, 14.95), (6.88, 14.95)])
    connector([(4.15, 13.58), (6.75, 13.58), (6.75, 13.90), (6.88, 13.90)])

    connector([(4.15, 12.03), (4.15, 11.62)])
    ax.plot([2.15, 8.05], [11.62, 11.62], color=COLORS["dark"], linewidth=1.15, zorder=0)
    connector([(2.15, 11.62), (2.15, 11.18)])
    connector([(5.00, 11.62), (5.00, 11.18)])
    connector([(8.05, 11.62), (8.05, 11.18)])

    connector([(5.00, 10.13), (5.00, 9.58)])
    ax.plot([3.65, 6.55], [9.58, 9.58], color=COLORS["dark"], linewidth=1.15, zorder=0)
    connector([(3.65, 9.58), (3.65, 9.08)])
    connector([(6.55, 9.58), (6.55, 9.08)])

    connector([(2.15, 10.13), (2.15, 7.20), (4.40, 7.20), (4.40, 6.90)])
    connector([(3.65, 8.03), (3.65, 7.45), (4.80, 7.45), (4.80, 6.90)])

    connector([(4.95, 5.83), (4.95, 5.18)])
    ax.plot([3.10, 6.90], [5.18, 5.18], color=COLORS["dark"], linewidth=1.15, zorder=0)
    connector([(3.10, 5.18), (3.10, 4.63)])
    connector([(6.90, 5.18), (6.90, 4.63)])
    connector([(3.10, 3.60), (3.10, 2.88)])

    fig.savefig(FIGURE_DIR / "fig1_PRISMA-flow diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_temporal_figure() -> None:
    plt, _, _ = require_matplotlib()
    summary = load_json(SUMMARY_JSON)
    years = sorted(int(y) for y in summary["years_all"].keys())
    annual = pd.DataFrame(
        {
            "Publication Year": years,
            "Articles": [summary["years_all"][str(y)] for y in years],
            "Metric-bearing abstracts": [summary["years_metric_bearing"][str(y)] for y in years],
        }
    )
    annual["Metric-bearing share"] = annual["Metric-bearing abstracts"] / annual["Articles"]
    save_csv(annual, "figure2_annual_counts_metric_share.csv")

    task_year = read_sheet("Tasks by Year").copy()
    save_csv(task_year, "figure2_task_by_year.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]})
    axes[0].plot(annual["Publication Year"], annual["Articles"], marker="o", color=COLORS["blue"], linewidth=2.5)
    axes[0].set_title("a  Annual corpus growth", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Articles")
    axes[0].set_xticks(years)
    axes[0].grid(axis="y", alpha=0.25)

    task_plot = task_year.set_index("Publication Year")
    cols = [c for c in task_plot.columns if c != "unassigned"]
    share = task_plot[cols].div(task_plot[cols].sum(axis=1), axis=0) * 100
    share.plot(kind="area", stacked=True, ax=axes[1], color=PALETTE[: len(cols)], linewidth=0)
    axes[1].set_title("b  Primary-task mix", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Assigned-task share (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].plot(
        annual["Publication Year"],
        annual["Metric-bearing share"] * 100,
        marker="o",
        color=COLORS["green"],
        linewidth=2.5,
    )
    axes[2].set_title("c  Abstract-level metric reporting", loc="left", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Metric-bearing abstracts (%)")
    axes[2].set_xticks(years)
    axes[2].set_ylim(50, 70)
    axes[2].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_temporal_tasks_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def barh(ax, labels: list[str], values: list[float], title: str, color: str, xlabel: str = "Articles") -> None:
    labels = [wrap(l, 24) for l in labels][::-1]
    values = values[::-1]
    ax.barh(labels, values, color=color)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def build_landscape_figure() -> None:
    plt, _, _ = require_matplotlib()
    cancer = read_sheet("Cancer Types Frequency").head(10)
    ai = read_sheet("AI Models Frequency").head(10)
    save_csv(cancer, "figure3_cancer_sites_top10.csv")
    save_csv(ai, "figure3_ai_families_top10.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    barh(
        axes[0],
        [clean_label(x) for x in cancer.iloc[:, 0]],
        [int(x) for x in cancer["Count"]],
        "a  Top cancer-site categories",
        COLORS["blue"],
    )
    barh(
        axes[1],
        [clean_label(x) for x in ai.iloc[:, 0]],
        [int(x) for x in ai["Count"]],
        "b  Top AI model families",
        COLORS["orange"],
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_disease_method_landscape.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_descriptor_dashboard_figure() -> None:
    plt, _, _ = require_matplotlib()
    descriptor_path = DATA_DIR / "table2_top_corpus_descriptors.csv"
    descriptors = pd.read_csv(descriptor_path)
    save_csv(descriptors, "figure3_top_corpus_descriptors.csv")

    panels = [
        ("a", "Primary tasks", "Primary task", COLORS["blue"], 0.32),
        ("b", "Cancer sites", "Cancer site", COLORS["green"], 0.43),
        ("c", "AI families", "AI family", COLORS["orange"], 0.37),
        ("d", "Composite metric sources", "Composite metric source", COLORS["purple"], 0.31),
    ]

    fig = plt.figure(figsize=(14, 8.2))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.04,
        right=0.98,
        top=0.86,
        bottom=0.10,
        wspace=0.12,
        hspace=0.22,
    )

    fig.text(0.04, 0.965, "Top corpus descriptors", ha="left", va="top", fontsize=24, fontweight="bold")
    fig.text(
        0.04,
        0.920,
        "Shares are percentages of the full 20,766-article corpus",
        ha="left",
        va="top",
        fontsize=13.5,
        color="#3f4552",
    )

    for idx, (letter, title, group, color, bar_x) in enumerate(panels):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        ax.set_axis_off()
        ax.set_xlim(0, 1)

        panel = descriptors[descriptors["Descriptor group"] == group].copy()
        panel["Descriptor"] = panel["Descriptor"].map(descriptor_label)
        max_count = float(panel["Count"].max())
        rows = len(panel)
        ax.set_ylim(-0.2, rows + 1.1)

        count_x = 0.80
        share_x = 0.985
        bar_max_width = count_x - bar_x - 0.06

        ax.text(0.01, rows + 0.65, letter, color=color, fontsize=18, fontweight="bold", va="center")
        ax.text(0.055, rows + 0.65, title, color=color, fontsize=15.5, fontweight="bold", va="center")
        ax.text(count_x, rows + 0.65, "Count", color=color, fontsize=10.5, ha="center", va="center")
        ax.text(share_x, rows + 0.65, "Share (%)", color=color, fontsize=10.5, ha="right", va="center")
        ax.hlines(rows + 0.25, 0.0, 0.99, color=color, linewidth=1.0)

        for row_idx, row in panel.reset_index(drop=True).iterrows():
            y = rows - row_idx - 0.35
            count = int(row["Count"])
            share = float(row["Share of full corpus (%)"])
            width = (count / max_count) * bar_max_width

            ax.text(0.01, y, row["Descriptor"], fontsize=10.8, ha="left", va="center", color="#151925")
            ax.barh(y, width, left=bar_x, height=0.48, color=color, edgecolor="none")
            ax.text(count_x, y, f"{count:,}", fontsize=11.2, ha="center", va="center", color="#151925")
            ax.text(share_x, y, f"{share:.2f}%", fontsize=11.2, ha="right", va="center", color="#151925")

    fig.add_artist(plt.Line2D([0.035, 0.98], [0.055, 0.055], color="#4b4f58", linewidth=0.8))
    fig.text(
        0.04,
        0.032,
        "Cancer-site and AI-family labels are multi-label descriptors and are not mutually exclusive.",
        ha="left",
        va="center",
        fontsize=10.8,
        color="#3f4552",
    )

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"fig3_top_corpus_descriptors.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_geography_figure() -> None:
    plt, _, _ = require_matplotlib()
    raw = read_sheet("Reprint Country Overall").head(10)
    per_capita = pd.read_excel(POPULATION_XLSX, sheet_name="PerCapita N100").head(10)
    translational = load_json(TRANSLATIONAL_JSON)
    trans_years = pd.DataFrame(
        sorted((int(k), v) for k, v in translational["high_confidence_years"].items()),
        columns=["Publication Year", "High-confidence translational subset"],
    )
    save_csv(raw, "figure4_country_raw_top10.csv")
    save_csv(per_capita, "figure4_country_per_capita_min100_top10.csv")
    save_csv(trans_years, "figure4_translational_subset_by_year.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), gridspec_kw={"width_ratios": [1.1, 1.15, 0.95]})
    barh(
        axes[0],
        [clean_label(x) for x in raw["Reprint-address country"]],
        [int(x) for x in raw["Count"]],
        "a  Raw corresponding-author output",
        COLORS["blue"],
    )
    barh(
        axes[1],
        [clean_label(x) for x in per_capita["Country"]],
        [float(x) for x in per_capita["Articles per 1M of population"]],
        "b  Per-capita output, at least 100 articles",
        COLORS["green"],
        xlabel="Articles per 1 million population",
    )
    axes[2].bar(
        trans_years["Publication Year"].astype(str),
        trans_years["High-confidence translational subset"],
        color=COLORS["purple"],
    )
    axes[2].set_title("c  Translational-signal subset", loc="left", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Articles")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig4_geography_translational.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    write_display_plan()
    build_table_data()
    build_flow_figure()
    build_temporal_figure()
    build_descriptor_dashboard_figure()
    build_geography_figure()
    print(f"Wrote visualization outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
