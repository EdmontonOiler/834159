import os
import sqlite3
from difflib import get_close_matches

import pandas as pd
import streamlit as st


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUTRITION_DB_PATH = os.path.join(APP_DIR, "module1_data", "nutrition_ingredients.sqlite")

NUTRIENT_FIELDS = [
    ("Energy", "energy_kj", "energy_kj_per_100g", "kJ", 0),
    ("Protein", "protein_g", "protein_g_per_100g", "g", 1),
    ("Fat, total", "fat_total_g", "fat_total_g_per_100g", "g", 1),
    ("Saturated fat", "fat_saturated_g", "fat_saturated_g_per_100g", "g", 1),
    ("Carbohydrate", "carbohydrate_g", "carbohydrate_g_per_100g", "g", 1),
    ("Sugars", "sugars_g", "sugars_g_per_100g", "g", 1),
    ("Sodium", "sodium_mg", "sodium_mg_per_100g", "mg", 0),
]

VALIDATION_SAMPLES = {
    "Yoghurt fruit cup": {
        "product_name": "Yoghurt fruit cup prototype",
        "final_weight": 335.0,
        "serving_size": 167.5,
        "package_weight": 335.0,
        "rows": [
            {
                "remove": False,
                "ingredient_name": "yoghurt, plain, unsweetened",
                "amount_g": 200.0,
                "selected_ingredient_id": "NZ_F57",
                "source": "NZ_CONCISE_2021",
            },
            {
                "remove": False,
                "ingredient_name": "banana, cavendish, peeled, raw",
                "amount_g": 80.0,
                "selected_ingredient_id": "AUS_F000262",
                "source": "AUSNUT_2023",
            },
            {
                "remove": False,
                "ingredient_name": "strawberry, raw",
                "amount_g": 40.0,
                "selected_ingredient_id": "AUS_F008952",
                "source": "AUSNUT_2023",
            },
            {
                "remove": False,
                "ingredient_name": "honey",
                "amount_g": 15.0,
                "selected_ingredient_id": "AUS_F004380",
                "source": "AUSNUT_2023",
            },
        ],
    },
    "Chicken rice bowl": {
        "product_name": "Chicken rice bowl prototype",
        "final_weight": 500.0,
        "serving_size": 250.0,
        "package_weight": 500.0,
        "rows": [
            {
                "remove": False,
                "ingredient_name": "rice, red, cooked",
                "amount_g": 220.0,
                "selected_ingredient_id": "AUS_F007660",
                "source": "AUSNUT_2023",
            },
            {
                "remove": False,
                "ingredient_name": "chicken, lean, raw",
                "amount_g": 160.0,
                "selected_ingredient_id": "AUS_F002691",
                "source": "AUSNUT_2023",
            },
            {
                "remove": False,
                "ingredient_name": "carrot, raw",
                "amount_g": 70.0,
                "selected_ingredient_id": "AUS_F002276",
                "source": "AUSNUT_2023",
            },
            {
                "remove": False,
                "ingredient_name": "broccoli, raw",
                "amount_g": 40.0,
                "selected_ingredient_id": "NZ_X1020",
                "source": "NZ_CONCISE_2021",
            },
            {
                "remove": False,
                "ingredient_name": "oil, olive",
                "amount_g": 10.0,
                "selected_ingredient_id": "AUS_F006177",
                "source": "AUSNUT_2023",
            },
        ],
    },
    "Blank recipe": {
        "product_name": "Untitled product",
        "final_weight": 100.0,
        "serving_size": 25.0,
        "package_weight": 100.0,
        "rows": [],
    },
}



st.set_page_config(page_title="Nutrition Information", layout="wide")

st.markdown(
    """
    <style>

    /* Sidebar navigation groups */
    [data-testid="stSidebarNav"] ul::before {
        content: "Risk Assessment";
        display: block;
        margin: 10px 18px 8px;
        font-size: 12px;
        font-weight: 900;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stSidebarNav"] ul li:nth-child(5)::before {
        content: "New Function";
        display: block;
        margin: 22px 18px 8px;
        font-size: 12px;
        font-weight: 900;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Navigation label fix: Streamlit names the entry script from app.py. */
    [data-testid="stSidebarNav"] ul li:first-child a span {
        display: none;
    }
    [data-testid="stSidebarNav"] ul li:first-child a::after {
        content: "App2.0";
        font-size: 16px;
        font-weight: 700;
        color: inherit;
    }

    .workspace-section-title {
        color: #0f172a;
        font-size: 18px;
        font-weight: 850;
        margin: 8px 0 12px;
    }
    .workspace-divider {
        height: 1px;
        background: #e5e7eb;
        margin: 20px 0 18px;
    }

    .sample-review-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        max-width: 680px;
        margin-top: -4px;
    }
    .sample-review-title {
        color: #0f172a;
        font-size: 14px;
        font-weight: 850;
        margin-bottom: 5px;
    }
    .sample-review-text,
    .sample-review-item {
        color: #475569;
        font-size: 13px;
        line-height: 1.42;
    }
    .sample-review-item {
        margin-top: 5px;
    }
    .sample-review-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 10px 0 9px;
    }

    .result-summary-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        margin: 6px 0 16px;
        padding: 10px 12px;
        border: 1px solid #dbeafe;
        border-radius: 10px;
        background: #f8fbff;
        color: #334155;
        font-size: 14px;
    }
    .result-summary-strip span {
        padding-right: 14px;
        border-right: 1px solid #dbe3ee;
    }
    .result-summary-strip span:last-child {
        border-right: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #f7f9fc;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 12px;
    }
    [data-testid="stSidebarNav"] a {
        border-radius: 10px;
        margin: 3px 10px;
        padding: 10px 12px;
        color: #334155;
        font-weight: 650;
        transition: all 0.18s ease;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: #eaf2ff;
        color: #1d4ed8;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #dbeafe;
        color: #1d4ed8;
        box-shadow: inset 3px 0 0 #2563eb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.4rem;
        max-width: 1280px;
    }
    .nutrition-hero {
        background:
            radial-gradient(circle at 92% 18%, rgba(255,255,255,0.16), transparent 28%),
            linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        color: white;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 14px;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.14);
    }
    .nutrition-hero h1 {
        margin: 0;
        font-size: 30px;
        line-height: 1.12;
        font-weight: 900;
        letter-spacing: 0;
    }
    .nutrition-hero p {
        margin: 8px 0 0 0;
        max-width: 920px;
        color: rgba(255,255,255,0.88);
        font-size: 14px;
        line-height: 1.45;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    .hero-badge {
        border: 1px solid rgba(255,255,255,0.26);
        background: rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 750;
    }
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 6px 0 14px 0;
    }
    .workflow-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 11px 14px;
        box-shadow: 0 5px 14px rgba(15, 23, 42, 0.035);
    }
    .workflow-step {
        color: #2563eb;
        font-size: 12px;
        font-weight: 850;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .workflow-title {
        color: #0f172a;
        font-size: 14px;
        font-weight: 850;
    }
    .workflow-text {
        color: #64748b;
        font-size: 12px;
        margin-top: 3px;
        line-height: 1.35;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: #dbe3ee !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    h3 {
        letter-spacing: 0;
        color: #0f172a;
    }
    .helper-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        margin-top: 4px;
    }
    .helper-title {
        color: #0f172a;
        font-weight: 850;
        font-size: 15px;
        margin-bottom: 7px;
    }
    .helper-item {
        color: #475569;
        font-size: 13px;
        margin: 8px 0;
        line-height: 1.35;
    }
    .result-banner {
        padding: 4px 2px 10px 2px;
        margin: 0 0 8px 0;
        border: none;
        background: transparent;
        box-shadow: none;
    }
    .result-banner-title {
        color: #0f172a;
        font-size: 22px;
        font-weight: 900;
        margin-bottom: 4px;
    }
    .result-banner-text {
        color: #64748b;
        font-size: 14px;
    }
    .stButton > button,
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #2563eb, #0f766e) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 9px 15px !important;
        font-weight: 750 !important;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.20) !important;
    }
    .stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.32) !important;
    }
    div[data-testid="column"]:has(button[kind="secondary"]) .stButton > button {
        width: 100%;
        min-height: 42px;
    }
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
    }
    .sample-button-caption {
        color: #64748b;
        font-size: 13px;
        margin-top: -2px;
        margin-bottom: 8px;
    }
    .blank-primary-button + div button {
        width: 100%;
        min-height: 46px;
        font-size: 15px !important;
        border-radius: 12px !important;
    }
    .sample-row-spacer {
        height: 10px;
    }
    .sample-mini-title {
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        margin: 0 0 6px 2px;
    }
    @media (max-width: 900px) {
        .workflow-grid {
            grid-template-columns: 1fr;
        }
        .nutrition-hero h1 {
            font-size: 30px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalise_ingredient_name(value):
    return " ".join(str(value).strip().lower().split())


def load_validation_sample(sample_name):
    sample = VALIDATION_SAMPLES[sample_name]
    st.session_state["product_name"] = sample["product_name"]
    st.session_state["final_weight"] = sample["final_weight"]
    st.session_state["serving_size"] = sample["serving_size"]
    st.session_state["package_weight"] = sample["package_weight"]
    st.session_state["recipe_rows"] = [row.copy() for row in sample["rows"]]
    st.session_state["recipe_editor_version"] = st.session_state.get("recipe_editor_version", 0) + 1
    st.session_state.pop("nutrition_result", None)


@st.cache_data
def load_nutrition_names():
    if not os.path.exists(NUTRITION_DB_PATH):
        return []

    conn = sqlite3.connect(NUTRITION_DB_PATH)
    try:
        aliases = pd.read_sql_query("SELECT alias FROM ingredient_aliases", conn)["alias"].tolist()
        names = pd.read_sql_query("SELECT standard_name FROM ingredients", conn)["standard_name"].tolist()
        return sorted(set(aliases + names))
    finally:
        conn.close()


def fetch_nutrition_match(input_name):
    normalised = normalise_ingredient_name(input_name)
    conn = sqlite3.connect(NUTRITION_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT
              a.alias,
              a.match_type,
              i.*
            FROM ingredient_aliases a
            JOIN ingredients i ON a.ingredient_id = i.ingredient_id
            WHERE a.alias = ?
            """,
            (normalised,),
        ).fetchone()

        if row is not None:
            return dict(row)

        row = conn.execute(
            "SELECT standard_name AS alias, 'exact' AS match_type, * FROM ingredients WHERE standard_name = ?",
            (normalised,),
        ).fetchone()

        if row is not None:
            return dict(row)

        return None
    finally:
        conn.close()




def get_nutrition_candidates(input_name, limit=6):
    normalised = normalise_ingredient_name(input_name)
    if not normalised or not os.path.exists(NUTRITION_DB_PATH):
        return []

    tokens = [token for token in normalised.replace(",", " ").split() if len(token) >= 3]
    conn = sqlite3.connect(NUTRITION_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        candidates = {}

        exact_rows = conn.execute(
            """
            SELECT
              i.ingredient_id,
              i.standard_name,
              i.category,
              i.energy_kj_per_100g,
              i.protein_g_per_100g,
              i.fat_total_g_per_100g,
              i.carbohydrate_g_per_100g,
              i.source,
              0 AS priority
            FROM ingredient_aliases a
            JOIN ingredients i ON a.ingredient_id = i.ingredient_id
            WHERE a.alias = ?
            UNION
            SELECT
              ingredient_id,
              standard_name,
              category,
              energy_kj_per_100g,
              protein_g_per_100g,
              fat_total_g_per_100g,
              carbohydrate_g_per_100g,
              source,
              0 AS priority
            FROM ingredients
            WHERE standard_name = ?
            """,
            (normalised, normalised),
        ).fetchall()

        for row in exact_rows:
            candidates[row["ingredient_id"]] = dict(row)

        like_patterns = [f"%{normalised}%"] + [f"%{token}%" for token in tokens[:3]]
        for idx, pattern in enumerate(like_patterns, start=1):
            rows = conn.execute(
                """
                SELECT
                  ingredient_id,
                  standard_name,
                  category,
                  energy_kj_per_100g,
                  protein_g_per_100g,
                  fat_total_g_per_100g,
                  carbohydrate_g_per_100g,
                  source,
                  ? AS priority
                FROM ingredients
                WHERE standard_name LIKE ?
                ORDER BY
                  CASE source
                    WHEN 'AUSNUT_2023' THEN 0
                    WHEN 'NZ_CONCISE_2021' THEN 1
                    WHEN 'prototype_seed' THEN 2
                    ELSE 3
                  END,
                  LENGTH(standard_name)
                LIMIT 20
                """,
                (idx, pattern),
            ).fetchall()

            for row in rows:
                candidates.setdefault(row["ingredient_id"], dict(row))

        def sort_key(row):
            source_rank = {
                "AUSNUT_2023": 0,
                "NZ_CONCISE_2021": 1,
                "prototype_seed": 2,
            }.get(row.get("source"), 3)
            return (row.get("priority", 9), source_rank, len(row.get("standard_name", "")))

        return sorted(candidates.values(), key=sort_key)[:limit]
    finally:
        conn.close()


def format_candidate_options(input_name, limit=5):
    options = get_nutrition_candidates(input_name, limit=limit)
    if not options:
        return ""
    return "; ".join(
        f"{row['standard_name']} [{row['source']}]"
        for row in options
    )



def fetch_nutrition_by_id(ingredient_id):
    if not ingredient_id or not os.path.exists(NUTRITION_DB_PATH):
        return None
    conn = sqlite3.connect(NUTRITION_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT standard_name AS alias, 'selected' AS match_type, * FROM ingredients WHERE ingredient_id = ?",
            (ingredient_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()



def render_nutrition_match_selectors(recipe_df, key_prefix):
    ingredient_options = {}
    edited_df = recipe_df.copy()
    records = edited_df.to_dict("records")
    entered = [
        (idx, str(row.get("ingredient_name", "")).strip())
        for idx, row in enumerate(records)
        if str(row.get("ingredient_name", "")).strip()
    ]
    if not entered:
        return edited_df

    with st.expander("Choose database match for each ingredient", expanded=False):
        st.caption(
            "Type an ingredient in the recipe table, then choose the closest database item here. "
            "The selected item will be used for nutrition calculation."
        )
        for idx, name in entered:
            candidates = get_nutrition_candidates(name, limit=8)
            if not candidates:
                st.warning(f"No close database option found for: {name}")
                ingredient_options[idx] = ""
                continue

            labels = [
                item["standard_name"]
                for item in candidates
            ]
            ids = [item["ingredient_id"] for item in candidates]
            default_index = 0
            current_selected = st.session_state.get(f"{key_prefix}_selected_id_{idx}", "")
            if current_selected in ids:
                default_index = ids.index(current_selected)

            choice = st.selectbox(
                f"{name}",
                options=labels,
                index=default_index,
                key=f"{key_prefix}_match_selector_{idx}_{name}",
            )
            selected_id = ids[labels.index(choice)]
            st.session_state[f"{key_prefix}_selected_id_{idx}"] = selected_id
            ingredient_options[idx] = selected_id

    selected_ids = []
    for idx in range(len(records)):
        selected_ids.append(ingredient_options.get(idx, ""))
    edited_df["selected_ingredient_id"] = selected_ids
    return edited_df

def calculate_nutrition_panel(recipe_df, final_product_weight_g, serving_size_g, package_weight_g):
    if "remove" in recipe_df.columns:
        recipe_df = recipe_df.drop(columns=["remove"])

    known_names = load_nutrition_names()
    totals = {total_key: 0.0 for _, total_key, _, _, _ in NUTRIENT_FIELDS}
    match_rows = []
    contribution_rows = []

    for _, row in recipe_df.iterrows():
        ingredient_name = str(row.get("ingredient_name", "")).strip()
        if not ingredient_name:
            continue

        amount_g = float(row.get("amount_g", 0) or 0)
        if amount_g <= 0:
            continue

        selected_ingredient_id = str(row.get("selected_ingredient_id", "") or "").strip()
        match = fetch_nutrition_by_id(selected_ingredient_id) if selected_ingredient_id else fetch_nutrition_match(ingredient_name)
        candidate_options = format_candidate_options(ingredient_name)
        contribution = {
            "input_name": ingredient_name,
            "amount_g": amount_g,
            "status": "unmatched",
            "standard_name": "",
        }

        if match:
            match_rows.append({
                "input_name": ingredient_name,
                "amount_g": amount_g,
                "status": "matched",
                "ingredient_id": match["ingredient_id"],
                "standard_name": match["standard_name"],
                "match_type": match["match_type"],
                "selected_ingredient_id": selected_ingredient_id,
                "source": match.get("source", ""),
                "candidate_options": candidate_options,
                "suggested_match": "",
                "note": "Used database nutrient values. Review candidate options if the ingredient is broad or ambiguous.",
            })
            contribution["status"] = "matched"
            contribution["standard_name"] = match["standard_name"]

            for _, total_key, db_key, _, _ in NUTRIENT_FIELDS:
                value = amount_g / 100 * float(match[db_key])
                totals[total_key] += value
                contribution[total_key] = round(value, 4)
        else:
            suggestions = get_close_matches(
                normalise_ingredient_name(ingredient_name),
                known_names,
                n=3,
                cutoff=0.72,
            )
            match_rows.append({
                "input_name": ingredient_name,
                "amount_g": amount_g,
                "status": "unmatched",
                "ingredient_id": "",
                "standard_name": "",
                "match_type": "",
                "source": "",
                "candidate_options": candidate_options,
                "suggested_match": "; ".join(suggestions),
                "note": "Confirm a suggested match, candidate option or enter manual nutrient data before final labelling.",
            })

            for _, total_key, _, _, _ in NUTRIENT_FIELDS:
                contribution[total_key] = None

        contribution_rows.append(contribution)

    nip_rows = []
    for label, total_key, _, unit, decimals in NUTRIENT_FIELDS:
        total_value = totals[total_key]
        per_100g = total_value / final_product_weight_g * 100 if final_product_weight_g else 0
        per_serving = total_value / final_product_weight_g * serving_size_g if final_product_weight_g else 0
        nip_rows.append({
            "Nutrient": label,
            "Quantity per serving": f"{per_serving:.0f} {unit}" if decimals == 0 else f"{per_serving:.1f} {unit}",
            "Quantity per 100 g": f"{per_100g:.0f} {unit}" if decimals == 0 else f"{per_100g:.1f} {unit}",
        })

    unmatched_count = sum(1 for row in match_rows if row["status"] == "unmatched")
    status = "Complete" if unmatched_count == 0 else "Partial - unmatched ingredients require review"
    servings_per_package = package_weight_g / serving_size_g if serving_size_g else 0

    return {
        "status": status,
        "unmatched_count": unmatched_count,
        "servings_per_package": servings_per_package,
        "nip": pd.DataFrame(nip_rows),
        "matches": pd.DataFrame(match_rows),
        "contributions": pd.DataFrame(contribution_rows),
    }


st.markdown(
    """
    <div class="nutrition-hero">
        <h1>Nutrition Information Generation</h1>
        <p>Generate a prototype Nutrition Information Panel from formulation data. Ingredient names are standardised using the local nutrition database before nutrient values are calculated per serving and per 100 g.</p>
        <div class="hero-badges">
            <div class="hero-badge">Ingredient matching</div>
            <div class="hero-badge">Per serving output</div>
            <div class="hero-badge">Per 100 g output</div>
            <div class="hero-badge">Missing ingredient review</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(NUTRITION_DB_PATH):
    st.error("Nutrition database not found. Please check module1_data/nutrition_ingredients.sqlite.")
    st.stop()

st.markdown(
    """
    <div class="workflow-grid">
        <div class="workflow-card">
            <div class="workflow-step">Step 1</div>
            <div class="workflow-title">Enter formulation</div>
            <div class="workflow-text">Add product weight, serving size and ingredient amounts.</div>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">Step 2</div>
            <div class="workflow-title">Standardise ingredients</div>
            <div class="workflow-text">Match user ingredient names against aliases in the database.</div>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">Step 3</div>
            <div class="workflow-title">Generate NIP</div>
            <div class="workflow-text">Calculate nutrition per serving and per 100 g for review.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "recipe_editor_version" not in st.session_state:
    st.session_state["recipe_editor_version"] = 0

required_sample_state = ["recipe_rows", "product_name", "final_weight", "serving_size", "package_weight"]
if any(key not in st.session_state for key in required_sample_state):
    load_validation_sample("Blank recipe")

with st.container(border=True):
    st.markdown("### Validation sample")
    sample_left, sample_right = st.columns([1.45, 2.55], gap="large")
    with sample_left:
        st.markdown(
            """
            <div style="font-size:14px;color:#0f172a;margin-bottom:7px;font-weight:600;">Load validation sample recipe</div>
            <div class="sample-button-caption">Use a sample, or start from a blank recipe.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="blank-primary-button"></div>', unsafe_allow_html=True)
        if st.button("Blank recipe", key="sample_blank_recipe"):
            load_validation_sample("Blank recipe")
            st.rerun()

        st.markdown('<div class="sample-row-spacer"></div>', unsafe_allow_html=True)
        sample_cols = st.columns(2)
        with sample_cols[0]:
            st.markdown('<div class="sample-mini-title">Sample</div>', unsafe_allow_html=True)
            if st.button("Yoghurt fruit cup", key="sample_yoghurt_fruit_cup"):
                load_validation_sample("Yoghurt fruit cup")
                st.rerun()
        with sample_cols[1]:
            st.markdown('<div class="sample-mini-title">Sample</div>', unsafe_allow_html=True)
            if st.button("Chicken rice bowl", key="sample_chicken_rice_bowl"):
                load_validation_sample("Chicken rice bowl")
                st.rerun()
    with sample_right:
        st.markdown(
            """
            <div class="sample-review-panel">
                <div class="sample-review-title">Sample purpose</div>
                <div class="sample-review-text">
                    Use a blank recipe for your own formulation, or load a sample recipe to check that
                    database matching, nutrient calculation and NIP generation are working correctly.
                </div>
                <div class="sample-review-divider"></div>
                <div class="sample-review-title">Review rules</div>
                <div class="sample-review-item">Search the database before adding each ingredient.</div>
                <div class="sample-review-item">Select the closest AUSNUT or NZ database item.</div>
                <div class="sample-review-item">Only confirmed ingredient records are used for nutrition calculation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    st.markdown("### Recipe formulation")

    st.markdown('<div class="workspace-section-title">Product information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        product_name = st.text_input("Product name", key="product_name")
    with c2:
        final_weight = st.number_input("Final product weight (g)", min_value=1.0, step=10.0, key="final_weight")
    with c3:
        serving_size = st.number_input("Serving size (g)", min_value=1.0, step=1.0, key="serving_size")
    with c4:
        package_weight = st.number_input("Package weight (g)", min_value=1.0, step=10.0, key="package_weight")

    st.markdown('<div class="workspace-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-section-title">Add ingredient from database</div>', unsafe_allow_html=True)
    search_left, match_mid, amount_right = st.columns([1.1, 1.8, 0.75], gap="medium")
    with search_left:
        search_term = st.text_input(
            "Search ingredient",
            key="module1_ingredient_search",
            placeholder="Type milk, honey, flour...",
        )

    candidates = get_nutrition_candidates(search_term, limit=12) if search_term else []
    selected_candidate = None
    with match_mid:
        if candidates:
            candidate_labels = [
                item["standard_name"]
                for item in candidates
            ]
            selected_label = st.selectbox(
                "Database match",
                options=candidate_labels,
                key="module1_database_match",
            )
            selected_candidate = candidates[candidate_labels.index(selected_label)]
        else:
            st.selectbox(
                "Database match",
                options=["Enter a search term to show database options"],
                disabled=True,
                key="module1_database_match_empty",
            )

    with amount_right:
        add_amount = st.number_input(
            "Amount (g)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="module1_add_amount",
        )

    add_col, hint_col = st.columns([0.8, 2.6])
    with add_col:
        add_selected = st.button("Add to recipe", key="module1_add_selected_ingredient")
    with hint_col:
        st.caption(
            "Search first, select the closest database item, enter the amount, then add it to the recipe."
        )

    if add_selected:
        if not selected_candidate:
            st.warning("Please search and choose a database ingredient before adding.")
        elif add_amount <= 0:
            st.warning("Please enter an amount greater than 0 g.")
        else:
            st.session_state["recipe_rows"].append({
                "remove": False,
                "ingredient_name": selected_candidate["standard_name"],
                "amount_g": float(add_amount),
                "selected_ingredient_id": selected_candidate["ingredient_id"],
                "source": selected_candidate.get("source", ""),
            })
            st.session_state["recipe_editor_version"] = st.session_state.get("recipe_editor_version", 0) + 1
            st.session_state.pop("nutrition_result", None)
            st.success(f"Added {selected_candidate['standard_name']} ({add_amount:.0f} g).")
            st.rerun()

    st.markdown('<div class="workspace-divider"></div>', unsafe_allow_html=True)
    selected_title, selected_action = st.columns([3.0, 1.0])
    with selected_title:
        st.markdown('<div class="workspace-section-title">Selected recipe ingredients</div>', unsafe_allow_html=True)
    with selected_action:
        remove_rows = st.button("Remove selected", key="remove_ingredient_rows")

    cleaned_recipe_rows = [
        row for row in st.session_state["recipe_rows"]
        if str(row.get("ingredient_name", "")).strip() or float(row.get("amount_g", 0) or 0) > 0
    ]
    st.session_state["recipe_rows"] = cleaned_recipe_rows
    recipe_source_df = pd.DataFrame(st.session_state["recipe_rows"])
    for column, default in {
        "remove": False,
        "ingredient_name": "",
        "amount_g": 0.0,
        "selected_ingredient_id": "",
        "source": "",
    }.items():
        if column not in recipe_source_df.columns:
            recipe_source_df[column] = default

    if remove_rows:
        remaining_rows = [
            row for row in recipe_source_df.to_dict("records")
            if not bool(row.get("remove", False))
        ]
        st.session_state["recipe_rows"] = remaining_rows
        st.session_state["recipe_editor_version"] = st.session_state.get("recipe_editor_version", 0) + 1
        st.session_state.pop("nutrition_result", None)
        st.rerun()

    if recipe_source_df.empty:
        recipe_source_df = pd.DataFrame(columns=["remove", "ingredient_name", "amount_g", "source", "selected_ingredient_id"])

    recipe_df = st.data_editor(
        recipe_source_df[["remove", "ingredient_name", "amount_g", "source", "selected_ingredient_id"]],
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "remove": st.column_config.CheckboxColumn("Remove", help="Select rows to delete"),
            "ingredient_name": st.column_config.TextColumn("Selected database ingredient"),
            "amount_g": st.column_config.NumberColumn("Amount (g)", min_value=0.0, step=1.0),
            "source": st.column_config.TextColumn("Source"),
            "selected_ingredient_id": st.column_config.TextColumn("Database ID"),
        },
        column_order=["remove", "ingredient_name", "amount_g", "source", "selected_ingredient_id"],
        disabled=["ingredient_name", "source", "selected_ingredient_id"],
        key=f"nutrition_recipe_editor_{st.session_state['recipe_editor_version']}",
    )
    st.session_state["recipe_rows"] = recipe_df.to_dict("records")

    st.markdown('<div class="workspace-divider"></div>', unsafe_allow_html=True)
    action_left, action_right = st.columns([1.1, 2.6], gap="large")
    with action_left:
        if st.button("Generate Nutrition Information Panel", key="generate_nip"):
            result = calculate_nutrition_panel(recipe_df, final_weight, serving_size, package_weight)
            st.session_state["nutrition_result"] = result
            st.session_state["nutrition_product_name"] = product_name
            st.session_state["nutrition_serving_size"] = serving_size
    with action_right:
        st.caption(
            "Only confirmed database selections are used for calculation. "
            "Any unexpected match should be reviewed before final labelling."
        )


if "nutrition_result" in st.session_state:
    result = st.session_state["nutrition_result"]
    with st.container(border=True):
        st.markdown(
            """
            <div class="result-banner">
                <div class="result-banner-title">Calculation results</div>
                <div class="result-banner-text">Review the generated NIP, ingredient matching status and nutrient contribution details.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        status_label = result["status"] if result["unmatched_count"] == 0 else "Review needed"
        st.markdown(
            f"""
            <div class="result-summary-strip">
                <span><strong>Status:</strong> {status_label}</span>
                <span><strong>Servings per package:</strong> {result["servings_per_package"]:.1f}</span>
                <span><strong>Unmatched ingredients:</strong> {result["unmatched_count"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["unmatched_count"] != 0:
            st.warning(result["status"])

        nip_tab, match_tab, contribution_tab = st.tabs([
            "Nutrition panel",
            "Ingredient matching",
            "Nutrient contributions",
        ])

        with nip_tab:
            st.dataframe(result["nip"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download NIP CSV",
                result["nip"].to_csv(index=False).encode("utf-8"),
                file_name="nutrition_information_panel.csv",
                mime="text/csv",
            )

        with match_tab:
            st.dataframe(result["matches"], use_container_width=True, hide_index=True)

        with contribution_tab:
            st.dataframe(result["contributions"], use_container_width=True, hide_index=True)
