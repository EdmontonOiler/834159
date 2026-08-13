import html
import os

import pandas as pd
import streamlit as st


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DOSE_PATH = os.path.join(APP_DIR, "module2_data", "allergen_reference_doses.csv")

SCENARIOS = {
    "Sample: Butter biscuit from shared line": {
        "allergen": "Peanut",
        "concentration": 20.0,
        "serving_size": 40.0,
        "description": "Low-risk multi-allergen example: several trace residues are detected from a shared biscuit manufacturing line.",
        "rows": [
            {"Allergen": "Peanut", "Detected concentration (mg/kg)": 20.0, "Serving size (g)": 40.0},
            {"Allergen": "Milk", "Detected concentration (mg/kg)": 12.0, "Serving size (g)": 40.0},
            {"Allergen": "Egg", "Detected concentration (mg/kg)": 10.0, "Serving size (g)": 40.0},
        ],
    },
    "Sample: Dark chocolate from shared line": {
        "allergen": "Milk",
        "concentration": 64.0,
        "serving_size": 25.0,
        "description": "Near-reference-dose multi-allergen example: milk is close to the reference dose while other residues remain lower.",
        "rows": [
            {"Allergen": "Milk", "Detected concentration (mg/kg)": 64.0, "Serving size (g)": 25.0},
            {"Allergen": "Peanut", "Detected concentration (mg/kg)": 25.0, "Serving size (g)": 25.0},
            {"Allergen": "Soy", "Detected concentration (mg/kg)": 80.0, "Serving size (g)": 25.0},
        ],
    },
    "Sample: Oat fruit bar from shared line": {
        "allergen": "Sesame",
        "concentration": 80.0,
        "serving_size": 40.0,
        "description": "Above-reference-dose multi-allergen example: sesame exceeds the reference dose and peanut or almond may also require review.",
        "rows": [
            {"Allergen": "Sesame", "Detected concentration (mg/kg)": 80.0, "Serving size (g)": 40.0},
            {"Allergen": "Peanut", "Detected concentration (mg/kg)": 55.0, "Serving size (g)": 40.0},
            {"Allergen": "Almond", "Detected concentration (mg/kg)": 12.0, "Serving size (g)": 40.0},
        ],
    },
    "Blank scenario": {
        "allergen": "Peanut",
        "concentration": 0.0,
        "serving_size": 40.0,
        "description": "Start from an empty user-defined scenario.",
        "rows": [
            {"Allergen": "Peanut", "Detected concentration (mg/kg)": 0.0, "Serving size (g)": 40.0},
        ],
    },
}


st.set_page_config(page_title="PAL Decision Support", layout="wide")

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

    .pal-help-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: -2px;
        margin-bottom: 8px;
    }
    .pal-help-panel .scenario-description-card {
        background: #eef4fb;
        min-height: 42px;
        padding: 10px 13px;
        font-size: 13px;
        line-height: 1.38;
    }
    .pal-help-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 9px 0 8px;
    }
    .pal-help-title {
        color: #0f172a;
        font-size: 14px;
        font-weight: 850;
        margin-bottom: 6px;
    }
    .pal-help-step {
        color: #475569;
        font-size: 12.5px;
        line-height: 1.38;
        margin-top: 4px;
    }
    .pal-input-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin: 14px 0 18px;
        padding: 11px 13px;
        background: #f8fafc;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        color: #475569;
        font-size: 13px;
        line-height: 1.35;
    }
    .pal-input-meta strong {
        color: #0f172a;
        font-weight: 800;
    }
    .pal-input-meta span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        white-space: nowrap;
    }
    .pal-input-meta span:not(:last-child)::after {
        content: "";
        width: 1px;
        height: 16px;
        margin-left: 8px;
        background: #cbd5e1;
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
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }
    .pal-hero {
        background:
            radial-gradient(circle at 90% 15%, rgba(255,255,255,0.22), transparent 28%),
            linear-gradient(135deg, #7c2d12 0%, #be123c 48%, #1d4ed8 100%);
        color: white;
        border-radius: 18px;
        padding: 30px 32px;
        margin-bottom: 20px;
        box-shadow: 0 18px 36px rgba(190, 18, 60, 0.18);
    }
    .pal-hero h1 {
        margin: 0;
        font-size: 36px;
        line-height: 1.12;
        font-weight: 900;
        letter-spacing: 0;
    }
    .pal-hero p {
        margin: 10px 0 0 0;
        max-width: 900px;
        color: rgba(255,255,255,0.88);
        font-size: 16px;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }
    .hero-badge {
        border: 1px solid rgba(255,255,255,0.30);
        background: rgba(255,255,255,0.14);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 750;
    }
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 20px 0;
    }
    .workflow-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .workflow-step {
        color: #be123c;
        font-size: 12px;
        font-weight: 850;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .workflow-title {
        color: #0f172a;
        font-size: 16px;
        font-weight: 850;
    }
    .workflow-text {
        color: #64748b;
        font-size: 13px;
        margin-top: 3px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border-color: #dbe3ee !important;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
    }
    .stButton > button,
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #be123c, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 11px 18px !important;
        font-weight: 750 !important;
        box-shadow: 0 8px 18px rgba(190, 18, 60, 0.22) !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px;
        line-height: 1.2;
        white-space: normal;
    }
    .decision-card {
        border-radius: 16px;
        padding: 18px 20px;
        margin-top: 14px;
        border: 1px solid #e5e7eb;
        background: white;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
    }
    .decision-title {
        font-size: 22px;
        font-weight: 900;
        margin-bottom: 6px;
        color: #0f172a;
    }
    .decision-text {
        color: #475569;
        font-size: 15px;
        line-height: 1.45;
    }
    .scenario-description-card {
        min-height: 52px;
        border-radius: 12px;
        background: #f1f5f9;
        color: #0f172a;
        padding: 13px 16px;
        border: 1px solid #e2e8f0;
        font-size: 15px;
        line-height: 1.35;
        display: flex;
        align-items: center;
    }
    .field-label {
        font-size: 14px;
        color: #0f172a;
        margin-bottom: 7px;
        font-weight: 600;
    }
    .sample-button-caption {
        color: #64748b;
        font-size: 13px;
        margin-top: -2px;
        margin-bottom: 8px;
    }
    .blank-primary-button + div button {
        width: 100%;
        min-height: 52px;
        font-size: 16px !important;
        border-radius: 14px !important;
    }
    .sample-row-spacer {
        height: 14px;
    }
    .sample-mini-title {
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        margin: 0 0 6px 2px;
    }
    .pal-compact-result-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin: 8px 0 18px 0;
        font-size: 13px;
    }
    .pal-compact-result-table th {
        background: #f8fafc;
        color: #64748b;
        font-weight: 750;
        text-align: left;
        padding: 10px 8px;
        border-bottom: 1px solid #e5e7eb;
        line-height: 1.2;
        white-space: normal;
    }
    .pal-compact-result-table td {
        color: #0f172a;
        padding: 10px 8px;
        border-bottom: 1px solid #eef2f7;
        vertical-align: middle;
        line-height: 1.25;
        word-break: normal;
        overflow-wrap: anywhere;
    }
    .pal-compact-result-table tr:last-child td {
        border-bottom: 0;
    }
    .pal-compact-result-table .num {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .pal-compact-result-table .allergen {
        width: 8%;
        font-weight: 800;
    }
    .pal-compact-result-table .small-col {
        width: 8%;
    }
    .pal-compact-result-table .medium-col {
        width: 10%;
    }
    .pal-compact-result-table .status-col {
        width: 13%;
    }
    .pal-compact-result-table .decision-col {
        width: 18%;
    }
    .pal-status-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: 800;
        line-height: 1.1;
    }
    .pal-status-pill.below {
        background: #dcfce7;
        color: #166534;
    }
    .pal-status-pill.above {
        background: #fee2e2;
        color: #991b1b;
    }
    .pal-guidance-text {
        font-weight: 650;
        color: #334155;
    }
    .pal-result-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
        gap: 14px;
        margin: 10px 0 20px 0;
    }
    .pal-result-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        background: #ffffff;
        padding: 16px 18px;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.045);
    }
    .pal-result-card.above {
        border-color: #fecaca;
        background: #fffafa;
    }
    .pal-result-card.below {
        border-color: #bbf7d0;
        background: #fbfffc;
    }
    .pal-result-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
    }
    .pal-result-card-title {
        color: #0f172a;
        font-size: 19px;
        font-weight: 850;
        line-height: 1.15;
    }
    .pal-result-badge {
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 12px;
        font-weight: 850;
        line-height: 1.15;
        text-align: center;
        white-space: normal;
        max-width: 185px;
    }
    .pal-result-badge.above {
        background: #dc2626;
        color: #ffffff;
    }
    .pal-result-badge.below {
        background: #16a34a;
        color: #ffffff;
    }
    .pal-result-details {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 9px 12px;
        margin: 10px 0 12px 0;
    }
    .pal-result-detail {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: rgba(248, 250, 252, 0.84);
        padding: 10px 11px;
        min-height: 62px;
    }
    .pal-result-detail-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
        line-height: 1.2;
        text-transform: uppercase;
        letter-spacing: 0.035em;
        margin-bottom: 5px;
    }
    .pal-result-detail-value {
        color: #0f172a;
        font-size: 16px;
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .pal-result-interpretation {
        border-top: 1px solid #e5e7eb;
        padding-top: 11px;
        color: #334155;
        font-size: 14px;
        line-height: 1.45;
    }
    .pal-result-interpretation strong {
        color: #0f172a;
    }
    @media (max-width: 760px) {
        .pal-result-card-grid {
            grid-template-columns: 1fr;
        }
        .pal-result-details {
            grid-template-columns: 1fr;
        }
    }
    .formula-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 18px 0;
    }
    .formula-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
    }
    .formula-title {
        color: #0f172a;
        font-size: 14px;
        font-weight: 850;
        margin-bottom: 6px;
    }
    .formula-body {
        color: #334155;
        font-size: 14px;
        line-height: 1.45;
    }
    .formula-code {
        font-family: Consolas, Monaco, monospace;
        color: #be123c;
        font-weight: 800;
    }
    .pal-summary-panel {
        margin-top: 18px;
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    .pal-summary-panel.required {
        border-color: #fecaca;
        background: #fff7f7;
    }
    .pal-summary-panel.review {
        border-color: #fde68a;
        background: #fffdf2;
    }
    .pal-summary-panel.low {
        border-color: #bbf7d0;
        background: #f7fff9;
    }
    .pal-summary-grid {
        display: grid;
        grid-template-columns: 1.3fr 0.9fr 0.75fr;
        gap: 12px;
        align-items: stretch;
    }
    .pal-summary-item {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .pal-summary-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 7px;
    }
    .pal-summary-value {
        color: #0f172a;
        font-size: 21px;
        font-weight: 850;
        line-height: 1.15;
    }
    .pal-summary-note {
        margin-top: 12px;
        color: #475569;
        font-size: 14px;
        line-height: 1.45;
    }
    .pal-download-spacer {
        height: 16px;
    }
    .pal-output-bottom-space {
        height: 18px;
    }
    @media (max-width: 900px) {
        .workflow-grid {
            grid-template-columns: 1fr;
        }
        .formula-grid {
            grid-template-columns: 1fr;
        }
        .pal-hero h1 {
            font-size: 30px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_reference_doses():
    reference_doses = pd.read_csv(REFERENCE_DOSE_PATH)
    reference_doses["source"] = (
        reference_doses["source"]
        .astype(str)
        .str.replace(" ED05", "", regex=False)
    )
    return reference_doses


def load_scenario(name):
    scenario = SCENARIOS[name]
    st.session_state["module2_allergen"] = scenario["allergen"]
    st.session_state["module2_allergen_select"] = scenario["allergen"]
    st.session_state["module2_concentration"] = scenario["concentration"]
    st.session_state["module2_serving_size"] = scenario["serving_size"]
    st.session_state["module2_description"] = scenario["description"]
    st.session_state["module2_assessment_rows"] = [
        {
            "Allergen": row.get("Allergen", scenario["allergen"]),
            "Detected concentration (mg/kg)": float(row.get("Detected concentration (mg/kg)", scenario["concentration"])),
            "Serving size (g)": float(row.get("Serving size (g)", scenario["serving_size"])),
        }
        for row in scenario.get("rows", [
            {
                "Allergen": scenario["allergen"],
                "Detected concentration (mg/kg)": scenario["concentration"],
                "Serving size (g)": scenario["serving_size"],
            }
        ])
    ]
    st.session_state.pop("module2_result", None)
    st.session_state.pop("module2_result_rows", None)


def calculate_exposure(concentration_mg_per_kg, serving_size_g, reference_dose_mg):
    exposure_mg = concentration_mg_per_kg * serving_size_g / 1000
    risk_ratio = exposure_mg / reference_dose_mg if reference_dose_mg else 0
    action_level_mg_per_kg = (reference_dose_mg * 1000 / serving_size_g) if serving_size_g else 0

    if concentration_mg_per_kg > action_level_mg_per_kg:
        decision = "Above action level - mitigation review needed"
        action_level_status = "Above action level"
        interpretation = (
            "The detected concentration is above the calculated action level. "
            "Additional risk mitigation should be reviewed. If the level cannot be reduced below the action level, PAL may be required."
        )
    else:
        decision = "At or below action level - PAL may not be required"
        action_level_status = "At or below action level"
        interpretation = (
            "The detected concentration is at or below the calculated action level. "
            "This supports omission of PAL, provided that the risk assessment and supporting evidence are documented."
        )

    return {
        "exposure_mg": exposure_mg,
        "risk_ratio": risk_ratio,
        "action_level_mg_per_kg": action_level_mg_per_kg,
        "action_level_status": action_level_status,
        "decision": decision,
        "interpretation": interpretation,
    }


def fmt_pal_number(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    formatted = f"{value:.{digits}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def render_pal_result_cards(display_df):
    for row in display_df.to_dict("records"):
        status = str(row.get("Action level status", ""))
        above_action_level = status.lower().startswith("above")
        badge_text = (
            "Above action level - mitigation review needed"
            if above_action_level
            else "At or below action level - PAL may not be required"
        )
        badge_color = "#dc2626" if above_action_level else "#16a34a"
        card_bg = "#fffafa" if above_action_level else "#fbfffc"
        card_border = "#fecaca" if above_action_level else "#bbf7d0"

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="border-left: 5px solid {card_border}; background: {card_bg}; padding: 2px 0 2px 10px; border-radius: 6px;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
                        <div style="font-size:22px; font-weight:850; color:#0f172a;">{html.escape(str(row.get('Allergen', '')))}</div>
                        <div style="background:{badge_color}; color:white; border-radius:999px; padding:7px 12px; font-size:13px; font-weight:850; max-width:360px; text-align:center;">
                            {html.escape(badge_text)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Detected concentration", f"{fmt_pal_number(row.get('Concentration (mg/kg)'), 2)} mg/kg food")
            c2.metric("Serving size", f"{fmt_pal_number(row.get('Serving size (g)'), 1)} g")
            c3.metric("Exposure per serving", f"{fmt_pal_number(row.get('Exposure per serving (mg protein)'), 3)} mg protein")

            c4, c5, c6 = st.columns(3)
            c4.metric("Reference dose", f"{fmt_pal_number(row.get('Reference dose (mg protein)'), 2)} mg protein")
            c5.metric("Action level", f"{fmt_pal_number(row.get('Action level (mg/kg food)'), 2)} mg/kg food")
            c6.metric("Risk ratio", fmt_pal_number(row.get("Risk ratio"), 2))

            st.markdown(
                f"""
                <div style="margin-top:8px; color:#334155; font-size:15px; line-height:1.5;">
                    <strong>Action level status:</strong> {html.escape(status)}<br>
                    <strong>Decision support:</strong> {html.escape(str(row.get('Decision support', '')))}<br>
                    <strong>Interpretation:</strong> {html.escape(str(row.get('Interpretation', '')))}
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_validation_summary(reference_doses):
    summary_rows = []
    sample_names = [name for name in SCENARIOS.keys() if name.startswith("Sample:")]

    for sample_name in sample_names:
        scenario = SCENARIOS[sample_name]
        allergen = scenario["allergen"]
        reference_row = reference_doses.loc[reference_doses["allergen"] == allergen].iloc[0]
        reference_dose = float(reference_row["reference_dose_mg_protein"])
        result = calculate_exposure(
            scenario["concentration"],
            scenario["serving_size"],
            reference_dose,
        )
        summary_rows.append({
            "Scenario": sample_name.replace("Sample: ", ""),
            "Allergen": allergen,
            "Concentration (mg/kg)": scenario["concentration"],
            "Serving size (g)": scenario["serving_size"],
            "Exposure (mg protein)": round(result["exposure_mg"], 4),
            "Reference dose (mg protein)": reference_dose,
            "Action level (mg/kg food)": round(result["action_level_mg_per_kg"], 2),
            "Action level status": result["action_level_status"],
            "Risk ratio": round(result["risk_ratio"], 2),
            "Decision support": result["decision"],
        })

    return pd.DataFrame(summary_rows)


if not os.path.exists(REFERENCE_DOSE_PATH):
    st.error("Reference dose table not found. Please check module2_data/allergen_reference_doses.csv.")
    st.stop()

reference_df = load_reference_doses()
allergen_options = reference_df["allergen"].tolist()

if "module2_allergen" not in st.session_state or "module2_assessment_rows" not in st.session_state:
    load_scenario("Blank scenario")

st.markdown(
    """
    <div class="pal-hero">
        <h1>Cross-Contact Allergen PAL Decision Support</h1>
        <p>Estimate allergen protein exposure per serving from detected concentration data, compare it with a selected VITAL 4.0 reference dose, and generate decision-support guidance for precautionary allergen labelling.</p>
        <div class="hero-badges">
            <div class="hero-badge">Exposure calculation</div>
            <div class="hero-badge">Reference dose comparison</div>
            <div class="hero-badge">Risk ratio</div>
            <div class="hero-badge">PAL decision support</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="workflow-grid">
        <div class="workflow-card">
            <div class="workflow-step">Step 1</div>
            <div class="workflow-title">Enter detection result</div>
            <div class="workflow-text">Provide allergen protein concentration in mg/kg food.</div>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">Step 2</div>
            <div class="workflow-title">Estimate exposure</div>
            <div class="workflow-text">Calculate mg allergen protein consumed per serving.</div>
        </div>
        <div class="workflow-card">
            <div class="workflow-step">Step 3</div>
            <div class="workflow-title">Support PAL decision</div>
            <div class="workflow-text">Compare the detected concentration with the calculated VITAL 4.0 action level.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### Validation scenario")
    sample_left, sample_right = st.columns([1.65, 2.2], gap="large")
    with sample_left:
        st.markdown(
            """
            <div class="field-label">Load validation scenario</div>
            <div class="sample-button-caption">Use a sample, or start from blank user input.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="blank-primary-button"></div>', unsafe_allow_html=True)
        if st.button("Blank user input", key="sample_blank_input"):
            load_scenario("Blank scenario")
            st.rerun()

        st.markdown('<div class="sample-row-spacer"></div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown('<div class="sample-mini-title">Sample</div>', unsafe_allow_html=True)
            if st.button("Butter biscuit", key="sample_peanut_biscuit"):
                load_scenario("Sample: Butter biscuit from shared line")
                st.rerun()
        with b2:
            st.markdown('<div class="sample-mini-title">Sample</div>', unsafe_allow_html=True)
            if st.button("Dark chocolate", key="sample_milk_chocolate"):
                load_scenario("Sample: Dark chocolate from shared line")
                st.rerun()
        with b3:
            st.markdown('<div class="sample-mini-title">Sample</div>', unsafe_allow_html=True)
            if st.button("Oat fruit bar", key="sample_sesame_granola"):
                load_scenario("Sample: Oat fruit bar from shared line")
                st.rerun()
    with sample_right:
        st.markdown(
            f"""
            <div class="pal-help-panel">
                <div class="field-label">Scenario description</div>
                <div class="scenario-description-card">
                    {st.session_state["module2_description"]}
                </div>
                <div class="pal-help-divider"></div>
                <div class="pal-help-title">How to use this module</div>
                <div class="pal-help-step"><strong>1.</strong> Load a sample or start from blank user input.</div>
                <div class="pal-help-step"><strong>2.</strong> Enter the detected allergen concentration and serving size.</div>
                <div class="pal-help-step"><strong>3.</strong> Calculate exposure and compare the detected concentration with the action level.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    st.markdown("### Risk assessment input")
    st.caption(
        "Add one row for each cross-contact allergen to be assessed. "
        "Detected concentration should be entered as mg allergen protein per kg food."
    )

    assessment_source_df = pd.DataFrame(st.session_state.get("module2_assessment_rows", []))
    for column, default in {
        "Allergen": allergen_options[0],
        "Detected concentration (mg/kg)": 0.0,
        "Serving size (g)": 40.0,
    }.items():
        if column not in assessment_source_df.columns:
            assessment_source_df[column] = default

    if assessment_source_df.empty:
        assessment_source_df = pd.DataFrame([
            {
                "Allergen": allergen_options[0],
                "Detected concentration (mg/kg)": 0.0,
                "Serving size (g)": 40.0,
            }
        ])

    assessment_input_df = st.data_editor(
        assessment_source_df[["Allergen", "Detected concentration (mg/kg)", "Serving size (g)"]],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Allergen": st.column_config.SelectboxColumn("Allergen", options=allergen_options, required=True),
            "Detected concentration (mg/kg)": st.column_config.NumberColumn(
                "Detected concentration (mg allergen protein / kg food)",
                min_value=0.0,
                step=0.1,
            ),
            "Serving size (g)": st.column_config.NumberColumn("Serving size (g)", min_value=0.1, step=1.0),
        },
        key="module2_multi_assessment_editor",
    )
    st.session_state["module2_assessment_rows"] = assessment_input_df.to_dict("records")

    st.markdown(
        """
        <div class="pal-input-meta">
            <span><strong>Reference:</strong> VITAL 4.0</span>
            <span><strong>Unit:</strong> mg allergen protein / kg food</span>
            <span><strong>Input:</strong> multiple cross-contact allergens</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Calculate PAL Decision Support", key="calculate_module2"):
        output_rows = []
        for row in assessment_input_df.to_dict("records"):
            allergen = row.get("Allergen")
            if allergen not in allergen_options:
                continue
            concentration = float(row.get("Detected concentration (mg/kg)", 0) or 0)
            serving_size = float(row.get("Serving size (g)", 0) or 0)
            reference_row = reference_df.loc[reference_df["allergen"] == allergen].iloc[0]
            reference_dose = float(reference_row["reference_dose_mg_protein"])
            result = calculate_exposure(concentration, serving_size, reference_dose)
            output_rows.append({
                "Allergen": allergen,
                "Concentration (mg/kg)": concentration,
                "Serving size (g)": serving_size,
                "Exposure per serving (mg protein)": result["exposure_mg"],
                "Reference dose (mg protein)": reference_dose,
                "Action level (mg/kg food)": result["action_level_mg_per_kg"],
                "Action level status": result["action_level_status"],
                "Risk ratio": result["risk_ratio"],
                "Decision support": result["decision"],
                "Reference source": reference_row["source"],
                "Interpretation": result["interpretation"],
            })

        st.session_state["module2_result_rows"] = output_rows
        st.session_state.pop("module2_result", None)

if "module2_result_rows" in st.session_state:
    output_df = pd.DataFrame(st.session_state["module2_result_rows"])

    with st.container(border=True):
        st.markdown("### Assessment output")
        st.caption("Review the PAL decision support result, exposure calculation and VITAL 4.0 reference-dose comparison.")

        if output_df.empty:
            st.warning("No valid allergen rows were available for calculation.")
        else:
            max_ratio = float(output_df["Risk ratio"].max())
            highest_row = output_df.sort_values("Risk ratio", ascending=False).iloc[0]
            if max_ratio > 1:
                overall_decision = "Mitigation review / PAL may be required"
                summary_class = "required"
                summary_note = (
                    f"Highest risk: {highest_row['Allergen']} has a risk ratio of {max_ratio:.2f}. "
                    "At least one allergen is above the calculated action level. Review whether additional mitigation can reduce the level below the action level before deciding on PAL."
                )
            else:
                overall_decision = "PAL may not be required"
                summary_class = "low"
                summary_note = (
                    f"Highest risk: {highest_row['Allergen']} has a risk ratio of {max_ratio:.2f}. "
                    "All assessed allergens are at or below their calculated action levels. Document and retain the risk assessment evidence supporting the omission of PAL."
                )

            display_df = output_df.copy()
            display_df["Exposure per serving (mg protein)"] = display_df["Exposure per serving (mg protein)"].round(4)
            display_df["Action level (mg/kg food)"] = display_df["Action level (mg/kg food)"].round(2)
            display_df["Risk ratio"] = display_df["Risk ratio"].round(2)

            result_tab, method_tab = st.tabs([
                "Multi-allergen assessment",
                "Calculation method",
            ])

            with result_tab:
                render_pal_result_cards(display_df)
            with method_tab:
                st.markdown(
                    """
                    <div class="formula-grid">
                        <div class="formula-card">
                            <div class="formula-title">Exposure per serving</div>
                            <div class="formula-body">
                                <span class="formula-code">Exposure = concentration x serving size / 1000</span><br>
                                Converts mg/kg food and serving size in g into mg allergen protein per serving.
                            </div>
                        </div>
                        <div class="formula-card">
                            <div class="formula-title">Action level</div>
                            <div class="formula-body">
                                <span class="formula-code">Action level = reference dose x 1000 / serving size</span><br>
                                Converts the VITAL 4.0 reference dose into a concentration limit for the selected serving size.
                            </div>
                        </div>
                        <div class="formula-card">
                            <div class="formula-title">Risk ratio</div>
                            <div class="formula-body">
                                <span class="formula-code">Risk ratio = exposure / reference dose</span><br>
                                This is equivalent to comparing detected concentration with the calculated action level.
                            </div>
                        </div>
                        <div class="formula-card">
                            <div class="formula-title">PAL decision workflow</div>
                            <div class="formula-body">
                                <span class="formula-code">Concentration <= action level: PAL may not be required</span><br>
                                <span class="formula-code">Concentration > action level: mitigation review / PAL may be required</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="pal-summary-panel {summary_class}">
                    <div class="pal-summary-grid">
                        <div class="pal-summary-item">
                            <div class="pal-summary-label">Overall decision support</div>
                            <div class="pal-summary-value">{overall_decision}</div>
                        </div>
                        <div class="pal-summary-item">
                            <div class="pal-summary-label">Highest risk allergen</div>
                            <div class="pal-summary-value">{highest_row['Allergen']}</div>
                        </div>
                        <div class="pal-summary-item">
                            <div class="pal-summary-label">Highest risk ratio</div>
                            <div class="pal-summary-value">{max_ratio:.2f}</div>
                        </div>
                    </div>
                    <div class="pal-summary-note">{summary_note}</div>
                </div>
                <div class="pal-download-spacer"></div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download assessment CSV",
                output_df.drop(columns=["Interpretation"]).to_csv(index=False).encode("utf-8"),
                file_name="pal_multi_allergen_decision_support_assessment.csv",
                mime="text/csv",
            )
            st.markdown('<div class="pal-output-bottom-space"></div>', unsafe_allow_html=True)

st.caption(
    "Decision-support note: this module estimates exposure using concentration and serving size data and VITAL 4.0 reference doses. "
    "It does not replace regulatory, clinical, analytical or food safety expert judgement."
)
