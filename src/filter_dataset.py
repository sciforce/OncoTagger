import os
import re
import time
from datetime import datetime
from collections import OrderedDict

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =========================================================
# BASIC SERVICE FUNCTIONS
# =========================================================

def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def uniq_lower(values):
    return list(
        OrderedDict.fromkeys(
            str(v).strip().lower()
            for v in values
            if pd.notna(v) and str(v).strip()
        )
    )


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower().strip()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def safe_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name in df.columns:
        return df[col_name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def derive_output_path(base_path: str, suffix: str) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".xlsx"
    return f"{root}{suffix}{ext}"


def split_wos_categories(text: str) -> list:
    if pd.isna(text) or not str(text).strip():
        return []
    return [part.strip().lower() for part in str(text).split(";") if part.strip()]


def compile_union_pattern(keywords: list[str]):
    keywords = uniq_lower(keywords)
    if not keywords:
        return None
    esc = sorted((re.escape(w) for w in keywords), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(esc) + r")(?!\w)", re.IGNORECASE)


def find_hits(text: str, pattern) -> list[str]:
    if not text or pattern is None:
        return []
    return list(OrderedDict.fromkeys(m.group(0).strip().lower() for m in pattern.finditer(text)))


def filter_hits_list(hits: list[str], ignore_terms: set[str] | None = None, custom_filter=None) -> list[str]:
    if not hits:
        return []
    ignore_terms = ignore_terms or set()
    out = []
    for h in hits:
        if h in ignore_terms:
            continue
        if custom_filter is not None and custom_filter(h):
            continue
        out.append(h)
    return out


def hits_to_string(bucket_hits: dict) -> str:
    parts = []
    for bucket in ["strong", "moderate", "weak", "remove"]:
        vals = bucket_hits.get(bucket, [])
        if vals:
            parts.append(f"{bucket}: " + "; ".join(vals))
    return " | ".join(parts)


def normalize_bucket_label(raw_label: str, default_bucket: str = "strong") -> str:
    label = normalize_text(raw_label)

    mapping = {
        "strong": "strong",
        "high": "strong",
        "primary": "strong",
        "moderate": "moderate",
        "medium": "moderate",
        "weak": "weak",
        "context": "weak",
        "contextual": "weak",
        "support": "weak",
        "supportive": "weak",
        "remove": "remove",
        "unsafe": "remove",
        "exclude": "remove",
        "non-ai": "remove",
        "non_ai": "remove",
        "non onco": "remove",
        "non-onco": "remove",
    }

    return mapping.get(label, default_bucket)


def load_tsv_set(tsv_path: str) -> set[str]:
    df = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str)
    return set(df.iloc[:, 0].dropna().astype(str).str.lower().str.strip().tolist())


def _read_csv_raw_flexible(csv_path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(
                csv_path,
                dtype=str,
                header=None,
                keep_default_na=False,
                encoding=encoding,
            )
        except Exception as exc:
            last_error = exc
    raise last_error


def _read_terms_csv_flexible(csv_path: str) -> pd.DataFrame:
    """
    Robust loader for both:
    1) CSV with header: term
    2) Headerless CSV with one column
    3) CSV with columns like term,bucket or keyword,class
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["term"])

    df = _read_csv_raw_flexible(csv_path).fillna("")
    if df.empty:
        return pd.DataFrame(columns=["term"])

    df.columns = [f"col_{i}" for i in range(df.shape[1])]
    first_row = [str(x).strip().lower() for x in df.iloc[0].tolist()]

    known_term_headers = {"term", "keyword", "phrase", "pattern"}
    known_bucket_headers = {"class", "bucket", "strength", "evidence", "tier", "group", "label"}

    has_header = False
    if first_row:
        if first_row[0] in known_term_headers:
            has_header = True
        elif len(first_row) > 1 and (first_row[0] in known_term_headers or first_row[1] in known_bucket_headers):
            has_header = True

    if has_header:
        new_cols = []
        for value in first_row:
            new_cols.append(value if value else f"unnamed_{len(new_cols)}")
        df = df.iloc[1:].copy()
        df.columns = new_cols
    else:
        if df.shape[1] == 1:
            df.columns = ["term"]
        elif df.shape[1] >= 2:
            df.columns = ["term", "bucket"] + [f"extra_{i}" for i in range(df.shape[1] - 2)]

    for col in df.columns:
        df[col] = df[col].astype(str).map(lambda x: x.strip())

    return df.fillna("")


def load_bucketed_terms(csv_path: str, default_bucket: str = "strong", fallback_remove_terms: list[str] | None = None):
    df = _read_terms_csv_flexible(csv_path)
    if df.empty:
        buckets = {"strong": [], "moderate": [], "weak": [], "remove": []}
        if fallback_remove_terms:
            buckets["remove"] = uniq_lower(fallback_remove_terms)
        patterns = {key: compile_union_pattern(vals) for key, vals in buckets.items()}
        return {"terms": buckets, "patterns": patterns}

    df.columns = [str(c).strip() for c in df.columns]
    buckets = {
        "strong": [],
        "moderate": [],
        "weak": [],
        "remove": [],
    }

    lower_cols = {c.lower(): c for c in df.columns}

    term_col = None
    for cand in ["term", "keyword", "phrase", "pattern"]:
        if cand in lower_cols:
            term_col = lower_cols[cand]
            break
    if term_col is None:
        term_col = df.columns[0]

    bucket_col = None
    for cand in ["class", "bucket", "strength", "evidence", "tier", "group", "label"]:
        if cand in lower_cols:
            bucket_col = lower_cols[cand]
            break

    if bucket_col is None:
        buckets[default_bucket] = uniq_lower(df[term_col].tolist())
    else:
        for _, row in df.iterrows():
            term = str(row[term_col]).strip()
            if not term:
                continue
            bucket = normalize_bucket_label(row[bucket_col], default_bucket=default_bucket)
            buckets[bucket].append(term)
        for key in buckets:
            buckets[key] = uniq_lower(buckets[key])

    if fallback_remove_terms:
        buckets["remove"] = uniq_lower(buckets["remove"] + fallback_remove_terms)

    patterns = {key: compile_union_pattern(vals) for key, vals in buckets.items()}
    return {"terms": buckets, "patterns": patterns}


def read_terms_csv_simple(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        return []
    df = _read_terms_csv_flexible(csv_path)
    if df.empty:
        return []
    first_col = df.columns[0]
    return uniq_lower(df[first_col].tolist())


def merge_bucket_stores(strong_terms, moderate_terms, weak_terms, remove_terms):
    buckets = {
        "strong": uniq_lower(strong_terms),
        "moderate": uniq_lower(moderate_terms),
        "weak": uniq_lower(weak_terms),
        "remove": uniq_lower(remove_terms),
    }
    patterns = {k: compile_union_pattern(v) for k, v in buckets.items()}
    return {"terms": buckets, "patterns": patterns}


def load_bucket_store_from_split_files(
    strong_csv: str,
    moderate_csv: str,
    weak_csv: str,
    remove_csv: str,
    fallback_single_csv: str | None = None,
    fallback_default_bucket: str = "strong",
    fallback_remove_terms: list[str] | None = None,
):
    strong_terms = read_terms_csv_simple(strong_csv)
    moderate_terms = read_terms_csv_simple(moderate_csv)
    weak_terms = read_terms_csv_simple(weak_csv)
    remove_terms = read_terms_csv_simple(remove_csv)

    split_mode_active = any([strong_terms, moderate_terms, weak_terms, remove_terms])

    if split_mode_active:
        if fallback_remove_terms:
            remove_terms = uniq_lower(remove_terms + fallback_remove_terms)
        return merge_bucket_stores(
            strong_terms=strong_terms,
            moderate_terms=moderate_terms,
            weak_terms=weak_terms,
            remove_terms=remove_terms,
        )

    if fallback_single_csv is None:
        raise FileNotFoundError("Neither split bucket CSV files nor fallback single CSV file were found.")

    return load_bucketed_terms(
        fallback_single_csv,
        default_bucket=fallback_default_bucket,
        fallback_remove_terms=fallback_remove_terms,
    )


def resolve_sources_dir(explicit_sources_dir: str | None = None) -> str:
    if explicit_sources_dir:
        return explicit_sources_dir

    base = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    candidates = [
        os.environ.get("FILTER_SOURCES_DIR", "").strip(),
        os.path.join(base, "sources"),
        os.path.normpath(os.path.join(base, os.pardir, "sources")),
        base,
        cwd,
        os.path.join(cwd, "sources"),
    ]

    needed_any = [
        "onco_terms_filter_strong.csv",
        "ai_terms_filter_strong.csv",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        if all(os.path.exists(os.path.join(candidate, name)) for name in needed_any):
            return candidate

    return os.path.normpath(os.path.join(base, os.pardir, "sources"))


def apply_with_progress(df: pd.DataFrame, func, desc: str, log_every: int = 1000):
    total = len(df)

    if tqdm is not None:
        tqdm.pandas(desc=desc)
        return df.progress_apply(func, axis=1)

    counter = {"n": 0, "started": time.time(), "last_log": time.time()}

    def wrapped(row):
        counter["n"] += 1
        n = counter["n"]
        now = time.time()

        if n == 1 or n % log_every == 0 or n == total or (now - counter["last_log"] >= 15):
            elapsed = now - counter["started"]
            rate = n / elapsed if elapsed > 0 else 0.0
            remaining = (total - n) / rate if rate > 0 else float("inf")
            log(
                f"{desc}: {n}/{total} "
                f"({n / total:.1%}) | elapsed={elapsed:.1f}s | "
                f"eta~{remaining:.1f}s"
            )
            counter["last_log"] = now

        return func(row)

    return df.apply(wrapped, axis=1)


# =========================================================
# INTERNAL DECONTAMINATION AND LOGIC CONSTANTS
# =========================================================
DEFAULT_AI_REMOVE = [
    "linear regression",
    "logistic regression",
    "multivariable regression",
    "multivariate regression",
    "multiple regression",
    "stepwise regression",
    "lasso regression",
    "ridge regression",
    "elastic net",
    "cox model",
    "cox regression",
    "proportional hazards model",
    "proportional hazards",
    "general linear model",
    "generalized linear model",
    "regression model",
    "statistical model",
    "survival model",
    "prognostic model",
    "prediction rule",
    "clinical prediction rule",
    "clinical prediction tool",
    "risk score",
    "risk model",
    "nomogram",
    "feature extraction",
    "regularization",
    "adam optimizer",
    "early stopping",
    "univariate analysis",
    "multivariate analysis",
]

DEFAULT_ONCO_REMOVE = [
    "lumpectomy",
    "mastectomy",
    "biopsy",
    "thyroid fine needle aspiration",
    "thyroid fine-needle aspiration",
    "breast magnetic resonance imaging",
    "breast mri",
    "american thyroid association",
    "hyperparathyroidism",
    "benign prostatic hyperplasia",
    "melanocytic nevus",
    "benign tumor",
    "benign tumour",
]

ONCO_OUTCOME_LEAK_TERMS = uniq_lower([
    "survival",
    "mortality",
    "prognosis",
    "overall survival",
    "progression-free survival",
    "progression free survival",
    "disease-free survival",
    "disease free survival",
    "recurrence-free survival",
    "recurrence free survival",
    "relapse-free survival",
    "relapse free survival",
    "cancer-specific survival",
    "cancer specific survival",
    "disease-specific survival",
    "disease specific survival",
])
ONCO_OUTCOME_LEAK_SET = set(ONCO_OUTCOME_LEAK_TERMS)

ONCO_OUTCOME_LEAK_PATTERNS = [
    re.compile(r"\bsurvival\b", re.I),
    re.compile(r"\bmortality\b", re.I),
    re.compile(r"\bprognos(?:is|tic)\b", re.I),
    re.compile(r"\bprogression[- ]free\b", re.I),
    re.compile(r"\bdisease[- ]free\b", re.I),
    re.compile(r"\brecurrence[- ]free\b", re.I),
    re.compile(r"\brelapse[- ]free\b", re.I),
]

def is_onco_outcome_leak_hit(hit: str) -> bool:
    return any(p.search(hit) for p in ONCO_OUTCOME_LEAK_PATTERNS)

WEAK_PREMALIGNANT_ONCO_TERMS = [
    "adenoma",
    "adenomas",
    "polyp",
    "polyps",
    "dysplasia",
    "barrett esophagus",
    "barrett's esophagus",
    "barrett oesophagus",
    "barrett's oesophagus",
    "nodule",
    "nodules",
    "lesion",
    "lesions",
    "nevus",
    "nevi",
]

GENERIC_BROAD_ONCOLOGY_TERMS = set(uniq_lower([
    "cancer",
    "cancers",
    "oncology",
    "oncologic",
    "oncological",
    "tumor",
    "tumors",
    "tumour",
    "tumours",
    "neoplasm",
    "neoplasms",
    "malignancy",
    "malignancies",
    "malignant neoplasm",
    "malignant neoplasms",
]))

ONCO_BACKGROUND_PATTERNS = [
    re.compile(r"\bcancer remains a (major|leading|significant|important) (burden|cause|problem)\b", re.I),
    re.compile(r"\bcancer is a (major|leading|significant|important) (burden|cause|problem)\b", re.I),
    re.compile(r"\b(global|worldwide|world) (cancer|oncology) burden\b", re.I),
    re.compile(r"\bthe burden of cancer\b", re.I),
    re.compile(r"\bcancer poses a (major|significant) public health (burden|problem)\b", re.I),
]

ONCO_EXCLUSION_PATTERNS = [
    re.compile(r"\b(excluded|exclude|excluding|exclusion criteria|ineligible|not eligible)\b.{0,160}\b(cancer|malignan\w+|tumou?r|neoplasm)\b", re.I),
    re.compile(r"\b(cancer|malignan\w+|tumou?r|neoplasm)\b.{0,160}\b(excluded|exclude|excluding|ineligible|not eligible)\b", re.I),
]

ONCO_COMORBIDITY_PATTERNS = [
    re.compile(
        r"\b(?:history of|prior history of|family history of|personal history of|past medical history of)\b.{0,160}\b(?:cancer|malignan\w+|tumou?r|neoplasm)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:comorbidit(?:y|ies)|comorbid condition(?:s)?|co-morbidit(?:y|ies))\b.{0,160}\b(?:cancer|malignan\w+|tumou?r|neoplasm)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:cancer|malignan\w+|tumou?r|neoplasm)\b.{0,160}\b(?:comorbidit(?:y|ies)|comorbid condition(?:s)?|co-morbidit(?:y|ies))\b",
        re.I,
    ),
]

NON_ONCO_DISEASE_TERMS = [
    "dementia",
    "alzheimer",
    "diabetes",
    "hypertension",
    "cardiovascular disease",
    "heart failure",
    "stroke",
    "kidney disease",
    "renal disease",
    "chronic kidney disease",
    "lung disease",
    "copd",
    "infection",
    "depression",
    "psychiatric",
    "sepsis",
    "obesity",
]
NON_ONCO_DISEASES_REGEX = "|".join(sorted((re.escape(x) for x in uniq_lower(NON_ONCO_DISEASE_TERMS)), key=len, reverse=True))

MULTI_DISEASE_LIST_PATTERNS = [
    re.compile(r"\b(diabetes|cardiovascular disease|stroke|kidney disease|lung disease|infection|dementia|hypertension|depression|obesity|cancer)\b(?:.{0,40},\s*|\s+and\s+|\s+or\s+){1,6}\b(diabetes|cardiovascular disease|stroke|kidney disease|lung disease|infection|dementia|hypertension|depression|obesity|cancer)\b", re.I),
    re.compile(r"\bincluding\b.{0,140}\b(cancer|diabetes|cardiovascular disease|stroke|infection|dementia|hypertension|depression|kidney disease)\b", re.I),
    re.compile(rf"\b(?:{NON_ONCO_DISEASES_REGEX})\b.{{0,120}}\b(cancer|malignan\w+|tumou?r|neoplasm)\b", re.I),
    re.compile(rf"\b(cancer|malignan\w+|tumou?r|neoplasm)\b.{{0,120}}\b(?:{NON_ONCO_DISEASES_REGEX})\b", re.I),
]

AI_GENERIC_WEAK_PATTERNS = [
    re.compile(r"\b(classification model|prediction model|predictive model|decision support system)\b", re.I),
    re.compile(r"\b(statistical model|risk model|clinical prediction tool|prognostic signature)\b", re.I),
]

LESION_AMBIGUITY_TERMS = [
    "lesion",
    "lesions",
    "nodule",
    "nodules",
    "mass",
    "masses",
    "focal lesion",
    "focal lesions",
]

ACTION_VERB_PATTERN = compile_union_pattern([
    "detect", "detects", "detected", "detection",
    "classify", "classifies", "classified", "classification",
    "diagnose", "diagnoses", "diagnostic", "diagnosis",
    "segment", "segments", "segmented", "segmentation",
    "predict", "predicts", "predicted", "prediction", "predictive",
    "forecast", "forecasts", "forecasting",
    "stratify", "stratification", "stratified",
    "screen", "screening",
    "risk prediction", "estimate risk", "risk stratification",
])

WEAK_PREMALIGNANT_ONCO_PATTERN = compile_union_pattern(WEAK_PREMALIGNANT_ONCO_TERMS)
LESION_AMBIGUITY_PATTERN = compile_union_pattern(LESION_AMBIGUITY_TERMS)

DIRECT_ONCOLOGY_CONTEXT_PATTERNS = [
    re.compile(r"\b(cancer|cancers|malignan\w+|tumou?rs?|neoplasms?|carcinoma(?:s)?) patients?\b", re.I),
    re.compile(r"\bpatients? with\b.{0,80}\b(cancer|cancers|malignan\w+|tumou?rs?|neoplasms?|carcinoma(?:s)?)\b", re.I),
    re.compile(r"\b(cancer|tumou?r|tumors|tumours|neoplasm|neoplasms|malignan\w+)\b.{0,24}\b(detection|diagnosis|screening|segmentation|classification)\b", re.I),
]

SITE_SPECIFIC_ONCOLOGY_PATTERNS = [
    re.compile(r"\b(?:breast|lung|pulmonary|colorectal|colon|rectal|gastric|stomach|pancreatic|pancreas|prostate|ovarian|ovary|endometrial|uterine|cervical|cervix|thyroid|renal|kidney|bladder|urothelial|hepatic|liver|hepatocellular|cholangio\w+|intrahepatic|esophageal|oesophageal|esophagus|oral|tongue|tonsil|head and neck|nasopharyngeal|laryngeal|brain|cns|glio\w+|melanoma|sarcoma|osteosarcoma|leukemia|leukaemia|lymphoma|myeloma|mesothelioma|neuroblastoma|medulloblastoma|retinoblastoma)\b.{0,24}\b(cancer|cancers|carcinoma|carcinomas|tumou?r|tumours|neoplasm|neoplasms|malignan\w+|sarcoma|lymphoma|leukemia|leukaemia|melanoma|myeloma|mesothelioma|blastoma)\b", re.I),
    re.compile(r"\b(glioblastoma|glioma|cholangiocarcinoma|hepatocellular carcinoma|ductal carcinoma in situ|dcis|triple negative breast cancer|non small cell lung cancer|small cell lung cancer|acute myeloid leukemia|acute lymphoblastic leukemia|diffuse large b-cell lymphoma|multiple myeloma|papillary thyroid carcinoma|intrahepatic cholangiocarcinoma|hepatoblastoma|medulloblastoma|neuroblastoma|retinoblastoma)\b", re.I),
]

MIXED_COHORT_PATTERNS = [
    re.compile(r"\b(mixed cohort|heterogeneous cohort|cancer and non-cancer|oncology and non-oncology|multiple diseases)\b", re.I),
]


# =========================================================
# HIT COLLECTION AND SIGNAL HELPERS
# =========================================================

def collect_bucket_hits(text: str, store: dict, ignore_terms: set[str] | None = None, custom_filter=None) -> dict:
    return {
        "strong": filter_hits_list(find_hits(text, store["patterns"]["strong"]), ignore_terms, custom_filter=custom_filter),
        "moderate": filter_hits_list(find_hits(text, store["patterns"]["moderate"]), ignore_terms, custom_filter=custom_filter),
        "weak": filter_hits_list(find_hits(text, store["patterns"]["weak"]), ignore_terms, custom_filter=custom_filter),
        "remove": filter_hits_list(find_hits(text, store["patterns"]["remove"]), ignore_terms, custom_filter=custom_filter),
    }


def has_primary_hits(bucket_hits: dict) -> bool:
    return bool(bucket_hits["strong"] or bucket_hits["moderate"])


def combine_hits(*hit_lists: list[str]) -> list[str]:
    out = []
    for lst in hit_lists:
        for item in lst:
            if item not in out:
                out.append(item)
    return out


def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def has_action_verbs_local(title_text: str, abstract_text: str) -> bool:
    if find_hits(title_text, ACTION_VERB_PATTERN):
        return True
    for sent in split_into_sentences(abstract_text):
        if find_hits(sent, ACTION_VERB_PATTERN):
            return True
    return False


def any_pattern_match(text: str, patterns: list) -> bool:
    return any(p.search(text) for p in patterns)


def has_non_generic_onco_primary(title_hits: dict, abstract_hits: dict) -> bool:
    all_primary = combine_hits(title_hits["strong"], title_hits["moderate"], abstract_hits["strong"], abstract_hits["moderate"])
    return any(hit not in GENERIC_BROAD_ONCOLOGY_TERMS and hit not in ONCO_OUTCOME_LEAK_SET for hit in all_primary)


def detect_site_specific_oncology_target(title_text: str, abstract_text: str) -> bool:
    text = f"{title_text} {abstract_text}".strip()
    return any_pattern_match(text, SITE_SPECIFIC_ONCOLOGY_PATTERNS)


def detect_direct_oncology_target(
    title_text: str,
    abstract_text: str,
    title_hits: dict,
    abstract_hits: dict,
    site_specific_oncology_target: bool,
    negative_only_context: bool,
    background_only_generic: bool,
) -> bool:
    primary_any = has_primary_hits(title_hits) or has_primary_hits(abstract_hits)
    if not primary_any:
        return False
    if site_specific_oncology_target:
        return True

    non_generic_primary = has_non_generic_onco_primary(title_hits, abstract_hits)
    direct_context = any_pattern_match(f"{title_text} {abstract_text}".strip(), DIRECT_ONCOLOGY_CONTEXT_PATTERNS)

    if background_only_generic and not non_generic_primary and not direct_context:
        return False
    if negative_only_context and not non_generic_primary and not direct_context:
        return False
    return bool(non_generic_primary or direct_context)


def detect_direct_ai_method(title_hits: dict, abstract_hits: dict) -> bool:
    return bool(title_hits["strong"] or title_hits["moderate"] or abstract_hits["strong"] or abstract_hits["moderate"])


# =========================================================
# CALCULATING HITS AND SCORING
# =========================================================

def score_oncology(row, onco_store: dict):
    title_text = row["title_text"]
    abstract_text = row["abstract_text"]
    keywords_text = row["keywords_text"]
    source_title_text = row["source_title_text"]
    text_combo = f"{title_text} {abstract_text}".strip()

    title_hits = collect_bucket_hits(title_text, onco_store, ignore_terms=ONCO_OUTCOME_LEAK_SET, custom_filter=is_onco_outcome_leak_hit)
    abstract_hits = collect_bucket_hits(abstract_text, onco_store, ignore_terms=ONCO_OUTCOME_LEAK_SET, custom_filter=is_onco_outcome_leak_hit)
    keywords_hits = collect_bucket_hits(keywords_text, onco_store, ignore_terms=ONCO_OUTCOME_LEAK_SET, custom_filter=is_onco_outcome_leak_hit)
    source_hits = collect_bucket_hits(source_title_text, onco_store, ignore_terms=ONCO_OUTCOME_LEAK_SET, custom_filter=is_onco_outcome_leak_hit)

    score = 0.0
    flags = []

    title_primary = has_primary_hits(title_hits)
    abstract_primary = has_primary_hits(abstract_hits)
    primary_any = title_primary or abstract_primary

    background_like = any_pattern_match(abstract_text[:600], ONCO_BACKGROUND_PATTERNS)
    exclusion_like = any_pattern_match(text_combo, ONCO_EXCLUSION_PATTERNS)
    comorbidity_like = any_pattern_match(text_combo, ONCO_COMORBIDITY_PATTERNS)
    multi_disease_like = any_pattern_match(text_combo, MULTI_DISEASE_LIST_PATTERNS)
    premalignant_like = bool(find_hits(title_text, WEAK_PREMALIGNANT_ONCO_PATTERN) or find_hits(abstract_text, WEAK_PREMALIGNANT_ONCO_PATTERN))
    lesion_like = bool(find_hits(title_text, LESION_AMBIGUITY_PATTERN) or find_hits(abstract_text, LESION_AMBIGUITY_PATTERN))

    raw_site_specific_oncology_target = detect_site_specific_oncology_target(title_text, abstract_text)
    non_generic_primary = has_non_generic_onco_primary(title_hits, abstract_hits)
    site_specific_oncology_target = bool(raw_site_specific_oncology_target and (non_generic_primary or not (comorbidity_like or multi_disease_like)))
    negative_only_context = bool(exclusion_like or comorbidity_like or multi_disease_like)
    background_only_generic = bool(background_like and not site_specific_oncology_target)
    direct_oncology_target = detect_direct_oncology_target(
        title_text=title_text,
        abstract_text=abstract_text,
        title_hits=title_hits,
        abstract_hits=abstract_hits,
        site_specific_oncology_target=site_specific_oncology_target,
        negative_only_context=negative_only_context,
        background_only_generic=background_only_generic,
    )

    if title_hits["strong"]:
        score += 4.0
    if abstract_hits["strong"]:
        score += 3.0
    if title_hits["moderate"]:
        score += 2.0
    if abstract_hits["moderate"]:
        score += 1.5

    if direct_oncology_target:
        if title_hits["weak"]:
            score += 0.60
        if abstract_hits["weak"]:
            score += 0.40
        if keywords_hits["strong"]:
            score += 0.40
        if keywords_hits["moderate"]:
            score += 0.25
        if keywords_hits["weak"]:
            score += 0.10
    elif primary_any:
        if title_hits["weak"]:
            score += 0.25
        if abstract_hits["weak"]:
            score += 0.15

    if source_hits["strong"] or source_hits["moderate"] or source_hits["weak"]:
        flags.append("onco_signal_in_source_title_only_supportive")

    if direct_oncology_target and has_action_verbs_local(title_text, abstract_text):
        score += 0.25

    if (title_hits["remove"] or abstract_hits["remove"] or keywords_hits["remove"]) and not direct_oncology_target:
        score -= 1.5
        flags.append("onco_unsafe_without_primary_onco")

    if premalignant_like and not direct_oncology_target:
        score -= 1.25
        flags.append("onco_premalignant_or_benign_ambiguity")

    if lesion_like and not direct_oncology_target:
        score -= 0.75
        flags.append("onco_lesion_like_ambiguity")

    if background_like and not direct_oncology_target and not site_specific_oncology_target:
        score -= 0.5
        flags.append("onco_background_like_mention")

    if exclusion_like:
        score -= 4.0
        flags.append("onco_exclusion_criterion_like_mention")

    if comorbidity_like:
        if direct_oncology_target:
            score -= 1.0
        else:
            score -= 3.5
        flags.append("onco_comorbidity_or_history_like_mention")

    if multi_disease_like:
        if site_specific_oncology_target:
            score -= 1.0
        elif direct_oncology_target:
            score -= 3.0
        else:
            score -= 4.0
        flags.append("onco_multi_disease_list_like_mention")

    if not direct_oncology_target:
        supportive_only = bool(title_hits["weak"] or abstract_hits["weak"] or keywords_hits["strong"] or keywords_hits["moderate"] or keywords_hits["weak"])
        if supportive_only or primary_any:
            flags.append("onco_no_clear_primary_target_signal")

    if site_specific_oncology_target:
        flags.append("site_specific_oncology_target")
    if direct_oncology_target:
        flags.append("direct_oncology_target")

    return {
        "score": round(score, 3),
        "title_hits": title_hits,
        "abstract_hits": abstract_hits,
        "keywords_hits": keywords_hits,
        "source_hits": source_hits,
        "primary_any": primary_any,
        "direct_oncology_target": direct_oncology_target,
        "site_specific_oncology_target": site_specific_oncology_target,
        "premalignant_ambiguity": bool(premalignant_like and not direct_oncology_target),
        "lesion_ambiguity": bool(lesion_like and not direct_oncology_target),
        "flags": uniq_lower(flags),
    }


def score_ai(row, ai_store: dict):
    title_text = row["title_text"]
    abstract_text = row["abstract_text"]
    keywords_text = row["keywords_text"]
    source_title_text = row["source_title_text"]
    text_combo = f"{title_text} {abstract_text}".strip()

    title_hits = collect_bucket_hits(title_text, ai_store)
    abstract_hits = collect_bucket_hits(abstract_text, ai_store)
    keywords_hits = collect_bucket_hits(keywords_text, ai_store)
    source_hits = collect_bucket_hits(source_title_text, ai_store)

    score = 0.0
    flags = []

    title_primary = has_primary_hits(title_hits)
    abstract_primary = has_primary_hits(abstract_hits)
    primary_any = title_primary or abstract_primary
    direct_ai_method = detect_direct_ai_method(title_hits, abstract_hits)

    generic_modeling_only = any_pattern_match(text_combo, AI_GENERIC_WEAK_PATTERNS) and not direct_ai_method
    non_ai_statistics_only = bool((title_hits["remove"] or abstract_hits["remove"] or keywords_hits["remove"]) and not direct_ai_method)

    if title_hits["strong"]:
        score += 4.0
    if abstract_hits["strong"]:
        score += 3.0
    if title_hits["moderate"]:
        score += 2.0
    if abstract_hits["moderate"]:
        score += 1.5

    if direct_ai_method:
        if title_hits["weak"]:
            score += 0.60
        if abstract_hits["weak"]:
            score += 0.40
        if keywords_hits["strong"]:
            score += 0.75
        if keywords_hits["moderate"]:
            score += 0.35
        if keywords_hits["weak"]:
            score += 0.10

    if source_hits["strong"] or source_hits["moderate"] or source_hits["weak"]:
        flags.append("ai_signal_in_source_title_only_supportive")

    if direct_ai_method and has_action_verbs_local(title_text, abstract_text):
        score += 0.25

    if non_ai_statistics_only:
        score -= 4.0
        flags.append("ai_non_ai_statistics_without_primary_ai")

    if generic_modeling_only:
        score -= 2.5
        flags.append("ai_generic_modeling_language_only")

    if not direct_ai_method and (keywords_hits["strong"] or keywords_hits["moderate"] or keywords_hits["weak"]):
        flags.append("ai_no_clear_primary_method_signal")

    if direct_ai_method:
        flags.append("direct_ai_method")

    return {
        "score": round(score, 3),
        "title_hits": title_hits,
        "abstract_hits": abstract_hits,
        "keywords_hits": keywords_hits,
        "source_hits": source_hits,
        "primary_any": primary_any,
        "direct_ai_method": direct_ai_method,
        "generic_modeling_only": generic_modeling_only,
        "non_ai_statistics_only": non_ai_statistics_only,
        "flags": uniq_lower(flags),
    }


def build_decision_reason(row) -> str:
    reasons = []
    reasons.append(f"decision={row['decision']}")
    reasons.append(f"oncology_score={row['oncology_score']}")
    reasons.append(f"ai_score={row['ai_score']}")
    reasons.append(f"direct_oncology_target={bool(row['direct_oncology_target'])}")
    reasons.append(f"site_specific_oncology_target={bool(row['site_specific_oncology_target'])}")
    reasons.append(f"direct_ai_method={bool(row['direct_ai_method'])}")
    reasons.append(f"severe_negative={bool(row['severe_negative'])}")

    if row["onco_flags"]:
        reasons.append(f"onco_flags={row['onco_flags']}")
    if row["ai_flags"]:
        reasons.append(f"ai_flags={row['ai_flags']}")

    severe_negative_reasons = row.get("severe_negative_reasons", "")
    if severe_negative_reasons:
        reasons.append(f"severe_negative_reasons={severe_negative_reasons}")

    if row["wos_exclusion_hits"]:
        reasons.append(f"wos_exclusion_hits={row['wos_exclusion_hits']}")

    return " | ".join(reasons)


def decide_record(row):
    onco_score = float(row["oncology_score"])
    ai_score = float(row["ai_score"])

    direct_oncology_target = bool(row["direct_oncology_target"])
    site_specific_oncology_target = bool(row["site_specific_oncology_target"])
    direct_ai_method = bool(row["direct_ai_method"])
    severe_negative = bool(row["severe_negative"])
    weak_oncology_ambiguity = bool(row["premalignant_ambiguity"] or row["lesion_ambiguity"] or row["mixed_cohort_ambiguity"])
    has_wos_exclusion = bool(str(row.get("wos_exclusion_hits", "")).strip())

    if severe_negative:
        return "exclude"

    if not direct_oncology_target or not direct_ai_method:
        if direct_ai_method and weak_oncology_ambiguity and onco_score >= 2.5 and ai_score >= 4.0 and not has_wos_exclusion:
            return "manual_review"
        return "exclude"

    if onco_score >= 4.0 and ai_score >= 4.5:
        return "include"

    if site_specific_oncology_target and onco_score >= 3.5 and ai_score >= 4.0:
        return "include"

    if weak_oncology_ambiguity and onco_score >= 3.0 and ai_score >= 4.0 and not has_wos_exclusion:
        return "manual_review"

    return "exclude"


# =========================================================
# MAIN FUNCTION
# =========================================================

def filter_dataset(
    input_file: str = "data/processed/processed_dataset.xlsx",
    output_file: str = "data/filtered/filtered_dataset.xlsx",
    review_output_file: str | None = None,
    excluded_output_file: str | None = None,
    audit_output_file: str | None = None,
    exclude_publication_years: tuple[str, ...] = ("2026",),
    sources_dir: str | None = None,
):
    started = time.time()

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    sources_dir = resolve_sources_dir(sources_dir)
    log(f"Input file: {input_file}")
    log(f"Resolved sources dir: {sources_dir}")

    excl_tsv = os.path.join(sources_dir, "wos_exclusion_categories.tsv")
    onco_strong_csv = os.path.join(sources_dir, "onco_terms_filter_strong.csv")
    onco_moderate_csv = os.path.join(sources_dir, "onco_terms_filter_moderate.csv")
    onco_weak_csv = os.path.join(sources_dir, "onco_terms_filter_weak.csv")
    onco_remove_csv = os.path.join(sources_dir, "onco_terms_filter_remove.csv")
    ai_strong_csv = os.path.join(sources_dir, "ai_terms_filter_strong.csv")
    ai_moderate_csv = os.path.join(sources_dir, "ai_terms_filter_moderate.csv")
    ai_weak_csv = os.path.join(sources_dir, "ai_terms_filter_weak.csv")
    ai_remove_csv = os.path.join(sources_dir, "ai_terms_filter_remove.csv")
    onco_legacy_csv = os.path.join(sources_dir, "onco_terms_filter.csv")
    ai_legacy_csv = os.path.join(sources_dir, "raw_ai_terms_filter.csv")

    if not os.path.exists(excl_tsv):
        raise FileNotFoundError(f"WoS exclusion TSV not found: {excl_tsv}")

    log("Loading dictionaries and regex patterns...")
    exclusion_categories = load_tsv_set(excl_tsv)

    onco_store = load_bucket_store_from_split_files(
        strong_csv=onco_strong_csv,
        moderate_csv=onco_moderate_csv,
        weak_csv=onco_weak_csv,
        remove_csv=onco_remove_csv,
        fallback_single_csv=onco_legacy_csv,
        fallback_default_bucket="strong",
        fallback_remove_terms=DEFAULT_ONCO_REMOVE,
    )

    ai_store = load_bucket_store_from_split_files(
        strong_csv=ai_strong_csv,
        moderate_csv=ai_moderate_csv,
        weak_csv=ai_weak_csv,
        remove_csv=ai_remove_csv,
        fallback_single_csv=ai_legacy_csv,
        fallback_default_bucket="strong",
        fallback_remove_terms=DEFAULT_AI_REMOVE,
    )

    log(
        "Loaded terms | "
        f"onco strong={len(onco_store['terms']['strong'])}, "
        f"moderate={len(onco_store['terms']['moderate'])}, "
        f"weak={len(onco_store['terms']['weak'])}, "
        f"remove={len(onco_store['terms']['remove'])} | "
        f"ai strong={len(ai_store['terms']['strong'])}, "
        f"moderate={len(ai_store['terms']['moderate'])}, "
        f"weak={len(ai_store['terms']['weak'])}, "
        f"remove={len(ai_store['terms']['remove'])}"
    )

    log("Reading Excel dataset...")
    df = pd.read_excel(input_file, dtype=str).fillna("")
    initial_count = len(df)
    log(f"Rows loaded: {initial_count}")

    log("Applying publication year filter...")
    df["Publication Year"] = safe_series(df, "Publication Year").map(normalize_text)
    mask_excluded_year = df["Publication Year"].isin({str(y).strip().lower() for y in exclude_publication_years})
    dropped_year = int(mask_excluded_year.sum())
    df = df.loc[~mask_excluded_year].copy()
    log(f"Dropped by year: {dropped_year} | Remaining: {len(df)}")

    log("Normalizing text fields...")
    df["title_text"] = safe_series(df, "Article Title").map(normalize_text)
    df["abstract_text"] = safe_series(df, "Abstract").map(normalize_text)
    df["author_keywords_text"] = safe_series(df, "Author Keywords").map(normalize_text)
    df["keywords_plus_text"] = safe_series(df, "Keywords Plus").map(normalize_text)
    df["keywords_text"] = (df["author_keywords_text"] + " " + df["keywords_plus_text"]).str.strip()
    df["source_title_text"] = safe_series(df, "Source Title").map(normalize_text)
    df["wos_categories_text"] = safe_series(df, "WoS Categories").map(normalize_text)
    df["wos_categories_list"] = safe_series(df, "WoS Categories").map(split_wos_categories)

    log("Applying WoS category trace layer...")
    df["wos_exclusion_hits"] = df["wos_categories_list"].apply(
        lambda cats: "; ".join([c for c in cats if c in exclusion_categories])
    )

    log("Scoring oncology relevance...")
    onco_results = apply_with_progress(df, lambda row: score_oncology(row, onco_store), desc="Oncology scoring")

    log("Scoring AI relevance...")
    ai_results = apply_with_progress(df, lambda row: score_ai(row, ai_store), desc="AI scoring")

    log("Materializing scores, flags, and gates...")
    df["oncology_score"] = onco_results.map(lambda x: x["score"])
    df["ai_score"] = ai_results.map(lambda x: x["score"])

    df["onco_primary_any"] = onco_results.map(lambda x: x["primary_any"])
    df["ai_primary_any"] = ai_results.map(lambda x: x["primary_any"])

    df["direct_oncology_target"] = onco_results.map(lambda x: x["direct_oncology_target"])
    df["site_specific_oncology_target"] = onco_results.map(lambda x: x["site_specific_oncology_target"])
    df["direct_ai_method"] = ai_results.map(lambda x: x["direct_ai_method"])

    df["premalignant_ambiguity"] = onco_results.map(lambda x: x["premalignant_ambiguity"])
    df["lesion_ambiguity"] = onco_results.map(lambda x: x["lesion_ambiguity"])
    df["mixed_cohort_ambiguity"] = (df["title_text"] + " " + df["abstract_text"]).map(lambda x: any_pattern_match(x, MIXED_COHORT_PATTERNS))

    df["onco_flags"] = onco_results.map(lambda x: "; ".join(x["flags"]))
    df["ai_flags"] = ai_results.map(lambda x: "; ".join(x["flags"]))

    df["onco_hits_title"] = onco_results.map(lambda x: hits_to_string(x["title_hits"]))
    df["onco_hits_abstract"] = onco_results.map(lambda x: hits_to_string(x["abstract_hits"]))
    df["onco_hits_keywords"] = onco_results.map(lambda x: hits_to_string(x["keywords_hits"]))
    df["onco_hits_source_title"] = onco_results.map(lambda x: hits_to_string(x["source_hits"]))

    df["ai_hits_title"] = ai_results.map(lambda x: hits_to_string(x["title_hits"]))
    df["ai_hits_abstract"] = ai_results.map(lambda x: hits_to_string(x["abstract_hits"]))
    df["ai_hits_keywords"] = ai_results.map(lambda x: hits_to_string(x["keywords_hits"]))
    df["ai_hits_source_title"] = ai_results.map(lambda x: hits_to_string(x["source_hits"]))

    log("Computing severe-negative gate...")
    df["severe_negative"] = False
    df.loc[df["onco_flags"].str.contains("onco_exclusion_criterion_like_mention", na=False), "severe_negative"] = True
    df.loc[df["ai_flags"].str.contains("ai_non_ai_statistics_without_primary_ai", na=False), "severe_negative"] = True
    df.loc[
        df["onco_flags"].str.contains("onco_comorbidity_or_history_like_mention", na=False)
        & ~df["direct_oncology_target"],
        "severe_negative",
    ] = True
    df.loc[
        df["onco_flags"].str.contains("onco_multi_disease_list_like_mention", na=False)
        & ~df["direct_oncology_target"],
        "severe_negative",
    ] = True
    df.loc[
        df["wos_exclusion_hits"].astype(str).str.strip().ne("")
        & ~df["direct_oncology_target"],
        "severe_negative",
    ] = True
    df.loc[
        df["wos_exclusion_hits"].astype(str).str.strip().ne("")
        & df["onco_flags"].str.contains(
            "onco_background_like_mention|onco_comorbidity_or_history_like_mention|onco_multi_disease_list_like_mention|onco_no_clear_primary_target_signal|onco_premalignant_or_benign_ambiguity|onco_lesion_like_ambiguity",
            regex=True,
            na=False,
        )
        & ~df["site_specific_oncology_target"],
        "severe_negative",
    ] = True
    df.loc[
        ~df["direct_ai_method"]
        & df["ai_flags"].str.contains("ai_generic_modeling_language_only|ai_no_clear_primary_method_signal", regex=True, na=False),
        "severe_negative",
    ] = True
    log(f"Severe-negative rows: {int(df['severe_negative'].sum())}")

    log(
        "Direct gates | "
        f"direct_oncology_target={int(df['direct_oncology_target'].sum())} | "
        f"site_specific_oncology_target={int(df['site_specific_oncology_target'].sum())} | "
        f"direct_ai_method={int(df['direct_ai_method'].sum())}"
    )
    
    log("Making final decisions...")
    df["decision"] = apply_with_progress(df, decide_record, desc="Decision stage")

    log("Building decision_reason column...")
    df["decision_reason"] = df.apply(build_decision_reason, axis=1)
    included_df = df[df["decision"] == "include"].copy()
    review_df = df[df["decision"] == "manual_review"].copy()
    excluded_df = df[df["decision"] == "exclude"].copy()

    log(
        f"Decision counts | include={len(included_df)} | "
        f"manual_review={len(review_df)} | exclude={len(excluded_df)}"
    )

    drop_tmp_cols = [
        "title_text",
        "abstract_text",
        "author_keywords_text",
        "keywords_plus_text",
        "keywords_text",
        "source_title_text",
        "wos_categories_text",
        "wos_categories_list",
    ]

    for frame in [included_df, review_df, excluded_df, df]:
        existing = [c for c in drop_tmp_cols if c in frame.columns]
        frame.drop(columns=existing, inplace=True, errors="ignore")

    if review_output_file is None:
        review_output_file = derive_output_path(output_file, "_manual_review")
    if excluded_output_file is None:
        excluded_output_file = derive_output_path(output_file, "_excluded")
    if audit_output_file is None:
        audit_output_file = derive_output_path(output_file, "_audit_all_decisions")

    ensure_parent_dir(output_file)
    ensure_parent_dir(review_output_file)
    ensure_parent_dir(excluded_output_file)
    ensure_parent_dir(audit_output_file)

    log("Saving Excel outputs...")
    included_df.to_excel(output_file, index=False)
    log(f"Saved include file: {output_file}")

    review_df.to_excel(review_output_file, index=False)
    log(f"Saved manual review file: {review_output_file}")

    excluded_df.to_excel(excluded_output_file, index=False)
    log(f"Saved exclude file: {excluded_output_file}")

    df.to_excel(audit_output_file, index=False)
    log(f"Saved audit file: {audit_output_file}")

    total_elapsed = time.time() - started
    print("=" * 70, flush=True)
    print("FILTERING SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial dataset size: {initial_count}", flush=True)
    print(f"Dropped by publication year: {dropped_year}", flush=True)
    print("-" * 70, flush=True)
    print(f"Included:       {len(included_df)}", flush=True)
    print(f"Manual review:  {len(review_df)}", flush=True)
    print(f"Excluded:       {len(excluded_df)}", flush=True)
    print(f"Total after year filter: {len(df)}", flush=True)
    print(f"Total elapsed seconds: {total_elapsed:.1f}", flush=True)
    print("=" * 70, flush=True)
    print(f"[OK] Included dataset saved to      -> {output_file}", flush=True)
    print(f"[OK] Manual-review dataset saved to -> {review_output_file}", flush=True)
    print(f"[OK] Excluded dataset saved to      -> {excluded_output_file}", flush=True)
    print(f"[OK] Full audit dataset saved to    -> {audit_output_file}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    filter_dataset()
