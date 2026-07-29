import pandas as pd
import streamlit as st


st.set_page_config(page_title="App2.0 Demo and Explanation", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #f7f9fc;
        border-right: 1px solid #e5e7eb;
    }
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
    [data-testid="stSidebarNav"] ul li:first-child a span {
        display: none;
    }
    [data-testid="stSidebarNav"] ul li:first-child a::after {
        content: "App2.0";
        font-size: 16px;
        font-weight: 700;
        color: inherit;
    }
    [data-testid="stSidebarNav"] a {
        border-radius: 10px;
        margin: 3px 10px;
        padding: 10px 12px;
        color: #334155;
        font-weight: 650;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #dbeafe;
        color: #1d4ed8;
        box-shadow: inset 3px 0 0 #2563eb;
    }
    .block-container {
        padding-top: 1.6rem;
        max-width: 1320px;
    }
    .demo-hero {
        padding: 26px 28px;
        border-radius: 16px;
        background: linear-gradient(135deg, #10264f, #1d4ed8);
        color: white;
        margin-bottom: 20px;
    }
    .demo-hero h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 900;
    }
    .demo-hero p {
        color: #dbeafe;
        margin: 10px 0 0;
        font-size: 16px;
    }
    .demo-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        background: #ffffff;
        margin-bottom: 18px;
    }
    .step-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .step-num {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: #1d4ed8;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
    }
    .step-text {
        font-size: 22px;
        font-weight: 900;
        color: #0f172a;
    }
    .mock-upload {
        border: 2px dashed #60a5fa;
        background: #f8fbff;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        color: #334155;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .mock-label {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        background: #f8fafc;
        padding: 14px;
        text-align: left;
        font-family: Consolas, monospace;
        font-size: 13px;
        color: #0f172a;
    }
    .pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #eef5ff;
        color: #1d4ed8;
        font-weight: 800;
        font-size: 13px;
        margin: 3px 4px 3px 0;
    }
    .risk-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        color: white;
        font-weight: 900;
        font-size: 13px;
    }
    .muted {
        color: #64748b;
        font-size: 14px;
    }

    .explain-box {
        border: 1px solid #dbeafe;
        background: #f8fbff;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
        color: #334155;
    }
    .input-summary {
        border: 1px solid #e5e7eb;
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 190px;
    }
    .input-summary h4 {
        margin: 0 0 10px;
        color: #0f172a;
        font-size: 17px;
        font-weight: 900;
    }
    .input-summary p {
        color: #334155;
        line-height: 1.55;
    }
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 18px 0 18px;
    }
    .flow-item {
        border: 1px solid #e5e7eb;
        background: #ffffff;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        font-weight: 850;
        color: #0f172a;
    }
    .flow-item span {
        display: block;
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        margin-top: 4px;
    }
    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        background: #ffffff;
        min-height: 150px;
    }
    .result-card.success {
        background: #ecfdf3;
        border-color: #bbf7d0;
    }
    .result-card.info {
        background: #eff6ff;
        border-color: #bfdbfe;
    }
    .result-title {
        font-size: 15px;
        color: #334155;
        font-weight: 850;
        margin-bottom: 12px;
    }
    .result-value {
        font-size: 26px;
        font-weight: 950;
        color: #0f172a;
    }
    .result-note {
        margin-top: 8px;
        color: #475569;
        font-size: 14px;
    }
    .risk-card-demo {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        background: #ffffff;
        margin-bottom: 14px;
    }
    .bar-track {
        width: 100%;
        height: 24px;
        border-radius: 999px;
        background: #e5e7eb;
        overflow: hidden;
        margin-top: 8px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: #16a34a;
        color: white;
        font-size: 13px;
        font-weight: 900;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .demo-mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 14px 0;
    }
    .demo-mini-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        background: #ffffff;
        min-height: 120px;
    }
    .demo-mini-card strong {
        display: block;
        color: #0f172a;
        font-size: 16px;
        margin-bottom: 8px;
    }
    .upload-panel {
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 18px;
        background: linear-gradient(180deg, #f8fbff, #ffffff);
        min-height: 240px;
    }
    .upload-icon {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        background: #dbeafe;
        color: #1d4ed8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 950;
        margin-bottom: 14px;
    }
    .review-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 16px;
        margin-bottom: 18px;
    }
    .review-card {
        border: 1px solid #e5e7eb;
        border-radius: 13px;
        background: #ffffff;
        padding: 14px 16px;
        min-height: 150px;
    }
    .review-card h4 {
        margin: 0 0 10px;
        color: #0f172a;
        font-size: 16px;
        font-weight: 900;
    }
    .review-text {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        color: #334155;
        line-height: 1.55;
        font-size: 14px;
        min-height: 82px;
    }
    .quality-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 14px 0 20px;
    }
    .quality-item {
        border: 1px solid #dbeafe;
        background: #f8fbff;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .quality-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 800;
    }
    .quality-value {
        color: #0f172a;
        font-size: 20px;
        font-weight: 950;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="demo-hero">
        <h1>App2.0 Demo and System Explanation</h1>
        <p>This page automatically demonstrates the full App2.0 workflow, from image upload to final allergen risk assessment output.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This is a guided demo page. It does not require manual buttons. The main App2.0 page is where users upload real images and run the live assessment."
)

with st.container(border=True):
    st.markdown(
        """
        <div class="step-title">
            <div class="step-num">1</div>
            <div class="step-text">Upload a food label image</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="explain-box">
            The user starts by uploading a food label image. App2.0 then prepares the image for OCR and allows the user
            to choose between a faster OCR engine and a more accurate OCR engine for complex labels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(
            """
            <div class="upload-panel">
                <div class="upload-icon">+</div>
                <div style="font-size:22px;font-weight:950;color:#0f172a;">Upload label image</div>
                <p style="color:#475569;line-height:1.55;">
                    In the live App2.0 page, the user uploads a PNG, JPG or JPEG image of the product ingredient label.
                </p>
                <span class="pill">PNG</span>
                <span class="pill">JPG</span>
                <span class="pill">JPEG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="mock-label">
            LOW FAT VANILLA FLAVOURED YOGHURT<br>
            INGREDIENTS: Skim milk, concentrated skim milk, water, sugar, cream (from milk),
            thickeners (1422 from maize, 1442 from maize), milk solids, gelatine, flavours,
            acidity regulators, enzyme (lactase), live cultures.<br><br>
            CONTAINS: Milk and milk products.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="demo-mini-grid">
                <div class="demo-mini-card">
                    <strong>Fast OCR</strong>
                    Tesseract is used for clear English labels when speed is preferred.
                </div>
                <div class="demo-mini-card">
                    <strong>Accurate OCR</strong>
                    EasyOCR is used for complex layouts or multilingual labels.
                </div>
                <div class="demo-mini-card">
                    <strong>Translation</strong>
                    Non-English OCR output can be translated before assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    st.markdown(
        """
        <div class="step-title">
            <div class="step-num">2</div>
            <div class="step-text">OCR and translation review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="explain-box">
            App2.0 separates the OCR output into reviewable fields. This helps the user confirm what the system extracted
            before the text is passed into compliance checking and model prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="quality-strip">
            <div class="quality-item"><div class="quality-label">OCR engine</div><div class="quality-value">EasyOCR</div></div>
            <div class="quality-item"><div class="quality-label">OCR quality</div><div class="quality-value">Good</div></div>
            <div class="quality-item"><div class="quality-label">Translation</div><div class="quality-value">Applied if needed</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="review-grid">
            <div class="review-card">
                <h4>Original OCR text</h4>
                <div class="review-text">
                LOW FAT VANILLA FLAVOURED YOGHURT INGREDIENTS: Skim milk, concentrated skim milk, water, sugar,
                cream (from milk), thickeners, milk solids, gelatine, flavours, enzyme (lactase), live cultures.
                CONTAINS: Milk and milk products.
                </div>
            </div>
            <div class="review-card">
                <h4>Extracted ingredient list</h4>
                <div class="review-text">
                Skim milk, concentrated skim milk, water, sugar, cream (from milk), thickeners, milk solids,
                gelatine, flavours, enzyme (lactase), live cultures.
                </div>
            </div>
            <div class="review-card">
                <h4>Translated ingredient list</h4>
                <div class="review-text">
                Skim milk, concentrated skim milk, water, sugar, cream from milk, thickeners, milk solids,
                gelatine, flavours, lactase enzyme, live cultures.
                </div>
            </div>
            <div class="review-card">
                <h4>Translated allergen statement</h4>
                <div class="review-text">
                Contains milk and milk products.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.container(border=True):
    st.markdown(
        """
        <div class="step-title">
            <div class="step-num">3</div>
            <div class="step-text">Confirm assessment input</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="explain-box">
            After OCR and translation, App2.0 asks the user to review the extracted text. The confirmed text below is passed into three checks:
            direct allergen detection, derivative ingredient evidence review and the Linear SVM risk-ranking model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        st.markdown(
            """
            <div class="input-summary">
                <h4>Ingredient list used for assessment</h4>
                <p>
                Skim milk, concentrated skim milk, water, sugar, cream (from milk), thickeners
                (1422 from maize, 1442 from maize), milk solids, gelatine, flavours,
                acidity regulators, enzyme (lactase), live cultures.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="input-summary">
                <h4>Allergen statement</h4>
                <p>Contains: Milk and milk products.</p>
                <h4 style="margin-top:18px;">Confirmed product context</h4>
                <p>Low fat vanilla flavoured yoghurt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="flow-grid">
            <div class="flow-item">1. Detect allergens<span>milk terms found</span></div>
            <div class="flow-item">2. Check declaration<span>milk declared</span></div>
            <div class="flow-item">3. Review derivatives<span>gelatine flagged</span></div>
            <div class="flow-item">4. Integrate review<span>model rank + evidence</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.container(border=True):
    st.markdown(
        """
        <div class="step-title">
            <div class="step-num">4</div>
            <div class="step-text">Integrated results dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="explain-box">
            App2.0 combines rule-based label compliance, derivative ingredient evidence and Linear SVM model ranking.
            The final review priority is not decided by the model score alone; supporting ingredient evidence can raise a result for supplier verification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="result-card success"><div class="result-title">Compliance</div><div class="result-value" style="color:#16a34a;">COMPLIANT</div><div class="result-note">All directly detected allergens are declared.</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="result-card success"><div class="result-title">Missing allergens</div><div class="result-value" style="color:#16a34a;">0</div><div class="result-note">None</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="result-card info"><div class="result-title">Detected allergens</div><div class="result-value">Milk</div><div class="result-note">Detected from skim milk, cream and milk solids.</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="result-card info"><div class="result-title">Declared allergens</div><div class="result-value">Milk</div><div class="result-note">Declared in the contains statement.</div></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="demo-mini-grid">
            <div class="result-card"><div class="result-title">High review</div><div class="result-value" style="color:#dc2626;">0</div><div class="result-note">No confirmed high derivative evidence.</div></div>
            <div class="result-card"><div class="result-title">Medium review</div><div class="result-value" style="color:#f59e0b;">1</div><div class="result-note">Fish source needs verification.</div></div>
            <div class="result-card"><div class="result-title">Low / monitor</div><div class="result-value" style="color:#16a34a;">7</div><div class="result-note">Model-ranked signals without supporting evidence.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Derivative ingredient evidence")
        st.markdown(
            """
            <div class="risk-card-demo">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                    <div style="font-size:20px;font-weight:950;color:#0f172a;">Fish</div>
                    <span class="risk-pill" style="background:#f59e0b;">Medium evidence</span>
                </div>
                <p style="margin-top:12px;color:#334155;"><strong>Matched terms:</strong> gelatine, gelatin</p>
                <p style="color:#64748b;">
                    Gelatine can be animal- or fish-derived. App2.0 flags it as a supplier-verification item rather than a confirmed undeclared allergen.
                </p>
                <div style="background:#f8fafc;border-radius:10px;padding:12px 14px;color:#0f172a;">
                    <strong>Recommended action:</strong> Verify raw material source with supplier and review allergen management documents.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "This section is decision support only. It flags indirect or derivative ingredient terms that may require supplier verification, not confirmed undeclared allergens."
        )

    with right:
        st.markdown("### Integrated Cross-Contact Allergen Risk Review")
        st.markdown(
            """
            <div class="explain-box">
                Relative risk score is a model ranking score from 0 to 100. The final review label combines this ranking with derivative evidence found in the ingredient text.
            </div>
            """,
            unsafe_allow_html=True,
        )

        integrated_results = [
            {"allergen": "Soy", "rank": 1, "score": 34.3, "label": "Low / monitor", "color": "#16a34a", "evidence": "No supporting derivative evidence found.", "reason": "This is a top-ranked model signal, but no supporting derivative evidence was found in the label text."},
            {"allergen": "Tree Nuts", "rank": 2, "score": 36.8, "label": "Low / monitor", "color": "#16a34a", "evidence": "No supporting derivative evidence found.", "reason": "This is a top-ranked model signal, but no supporting derivative evidence was found in the label text."},
            {"allergen": "Egg", "rank": 3, "score": 28.1, "label": "Low / monitor", "color": "#16a34a", "evidence": "No supporting derivative evidence found.", "reason": "This is a top-ranked model signal, but no supporting derivative evidence was found in the label text."},
            {"allergen": "Fish", "rank": 7, "score": 19.3, "label": "Medium review", "color": "#f59e0b", "evidence": "fish: gelatine, gelatin", "reason": "Derivative ingredient evidence was detected. Review ingredient source or supplier documentation."},
        ]

        for item in integrated_results:
            width = max(4, min(100, item["score"]))
            st.markdown(
                f"""
                <div class="risk-card-demo">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                        <div style="font-size:17px;font-weight:900;color:#0f172a;">{item['allergen']} <span style="font-size:12px;color:#64748b;margin-left:8px;">Model rank #{item['rank']}</span></div>
                        <span class="risk-pill" style="background:{item['color']};">{item['label']}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:14px;color:#64748b;font-size:14px;">
                        <span>Relative risk score</span>
                        <span>{item['score']:.1f}/100</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{width:.1f}%;background:{item['color']};">{item['score']:.1f}</div>
                    </div>
                    <p style="margin-top:12px;color:#334155;"><strong>Evidence:</strong> {item['evidence']}<br><strong>Reason:</strong> {item['reason']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="demo-card">
            <strong>How users should read this result:</strong>
            Milk is compliant because it is directly detected and declared. Fish is raised to Medium review because gelatine may be fish-derived and should be checked with the supplier.
            Other allergens are shown as model-ranked signals, but they remain Low / monitor when the label text provides no supporting derivative evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.caption(
    "This demo is illustrative. Real App2.0 results depend on the uploaded label image, OCR quality, translated text, Linear SVM ranking scores and derivative-evidence matching."
)
