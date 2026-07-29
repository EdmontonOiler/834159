import streamlit as st

st.set_page_config(page_title="What's New", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] ul li:first-child a span {
        display: none;
    }
    [data-testid="stSidebarNav"] ul li:first-child a::after {
        content: "App2.0";
        font-size: 16px;
        font-weight: 700;
        color: inherit;
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
    .block-container {
        max-width: 1280px;
        padding-top: 1.7rem;
    }
    .wn-hero {
        padding: 30px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, #10264f, #1d4ed8);
        color: white;
        margin-bottom: 22px;
    }
    .wn-hero h1 {
        margin: 0;
        font-size: 40px;
        font-weight: 950;
    }
    .wn-hero p {
        margin: 10px 0 0;
        color: #dbeafe;
        font-size: 16px;
        max-width: 860px;
        line-height: 1.55;
    }
    .section-title {
        font-size: 26px;
        font-weight: 950;
        color: #0f172a;
        margin: 24px 0 12px;
    }
    .wn-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px 24px;
        background: #ffffff;
        margin: 14px 0 20px;
        min-height: 150px;
    }
    .wn-card h3 {
        margin: 0 0 10px;
        color: #0f172a;
        font-size: 20px;
        font-weight: 950;
    }
    .wn-card p, .wn-card li {
        color: #334155;
        line-height: 1.58;
        font-size: 15px;
    }
    .wn-muted {
        color: #64748b;
        font-size: 14px;
    }
    .wn-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: #eef5ff;
        color: #1d4ed8;
        font-weight: 850;
        font-size: 13px;
        margin: 3px 4px 3px 0;
    }
    .metric-card {
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 18px;
        background: #f8fbff;
        margin-bottom: 14px;
    }
    .section-spacer {
        height: 10px;
    }
    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        margin-top: 8px;
        color: #0f172a;
        font-size: 30px;
        font-weight: 950;
    }
    .metric-note {
        margin-top: 5px;
        color: #475569;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wn-hero">
        <h1>What's New in App2.0</h1>
        <p>
            App2.0 upgrades the original allergen risk assessment prototype with a larger AU/NZ-focused dataset,
            optimized feature engineering, Linear SVM model selection, improved OCR review and an integrated
            cross-contact allergen risk review workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">1. Dataset Upgrade</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-label">Training rows</div><div class="metric-value">5,170</div><div class="metric-note">Final AU/NZ dataset</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-label">Positive samples</div><div class="metric-value">3,670</div><div class="metric-note">Products with risk labels</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-label">Negative samples</div><div class="metric-value">1,500</div><div class="metric-note">Products without risk labels</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-label">Risk labels</div><div class="metric-value">13</div><div class="metric-note">Multi-label prediction</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="wn-card">
        <h3>From WWS data to AU/NZ Open Food Facts data</h3>
        <p>
            App1.0 used the original WWS dataset. App2.0 introduces a new training dataset prepared from
            Open Food Facts product records, filtered toward Australia and New Zealand products so the model
            better reflects the regional food-label context used in this project.
        </p>
        <p>
            The final dataset keeps all usable positive examples and adds selected negative examples, allowing the
            model to learn both products where allergen-risk labels are present and products where they are absent.
        </p>
        <span class="wn-pill">Open Food Facts</span>
        <span class="wn-pill">Australia + New Zealand</span>
        <span class="wn-pill">Positive + negative samples</span>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("### Label Coverage")
st.dataframe(
    [
        {"Risk label": "Tree nuts", "Positive examples": 1921},
        {"Risk label": "Peanut", "Positive examples": 1430},
        {"Risk label": "Soy", "Positive examples": 1324},
        {"Risk label": "Milk", "Positive examples": 1203},
        {"Risk label": "Sesame", "Positive examples": 1109},
        {"Risk label": "Gluten", "Positive examples": 930},
        {"Risk label": "Egg", "Positive examples": 775},
        {"Risk label": "Lupin", "Positive examples": 319},
        {"Risk label": "Fish", "Positive examples": 316},
        {"Risk label": "Sulphites", "Positive examples": 307},
        {"Risk label": "Crustacean", "Positive examples": 176},
        {"Risk label": "Mollusc", "Positive examples": 44},
        {"Risk label": "Wheat", "Positive examples": 14},
    ],
    use_container_width=True,
    hide_index=True,
)

st.markdown('<div class="section-title">2. Model and Feature Upgrade</div>', unsafe_allow_html=True)

st.markdown("### Feature Sets Used for Comparison")
st.markdown(
    """
    <div class="wn-card">
        <p>
            The feature-set experiment used the same AU/NZ dataset, the same 75/25 train-test split and the same
            Linear SVM model. The only difference was the input feature set. Feature Set C was selected because it
            gave the strongest result with the final selected model.
        </p>
        <ul>
            <li><strong>A: Ingredient list only</strong> uses only product ingredient text as the baseline.</li>
            <li><strong>B: Ingredient + category</strong> adds product category information.</li>
            <li><strong>C: Optimized expanded features</strong> adds selected label-context signals for App2.0 prediction.</li>
        </ul>
        <span class="wn-pill">A = baseline</span>
        <span class="wn-pill">B = category added</span>
        <span class="wn-pill">C = App2.0 default</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Features Used in App2.0 Feature Set C")
st.dataframe(
    [
        {"Feature": "ingredient_list", "Purpose": "Primary ingredient text used to detect allergen-related terms and context."},
        {"Feature": "product_category", "Purpose": "Adds food type information, because allergen-risk patterns can vary by category."},
        {"Feature": "contains_statement", "Purpose": "Uses declared allergen text to understand what is already stated on the label."},
        {"Feature": "category_group", "Purpose": "Groups detailed product categories into broader food groups."},
        {"Feature": "ingredient_allergen_keyword_count", "Purpose": "Counts direct allergen keyword signals in the ingredient list."},
        {"Feature": "has_may_contain", "Purpose": "Identifies whether a precautionary allergen statement is present."},
    ],
    use_container_width=True,
    hide_index=True,
)

st.markdown("### Feature Set Performance Comparison")
st.dataframe(
    [
        {
            "Feature set": "A: Ingredient list only",
            "Model": "Linear SVM",
            "Micro F1": 0.664848,
            "Macro F1": 0.587709,
            "Micro recall": 0.673244,
            "Hamming loss": 0.110853,
            "Exact match accuracy": 0.387471,
        },
        {
            "Feature set": "B: Ingredient + category",
            "Model": "Linear SVM",
            "Micro F1": 0.665370,
            "Macro F1": 0.590819,
            "Micro recall": 0.674822,
            "Hamming loss": 0.110853,
            "Exact match accuracy": 0.383604,
        },
        {
            "Feature set": "C: Optimized expanded features",
            "Model": "Linear SVM",
            "Micro F1": 0.730539,
            "Macro F1": 0.638981,
            "Micro recall": 0.746251,
            "Hamming loss": 0.089907,
            "Exact match accuracy": 0.492653,
        },
    ],
    use_container_width=True,
    hide_index=True,
)

st.info(
    "This comparison keeps the dataset, split and model algorithm fixed. The result shows that Feature Set C improves the Linear SVM model compared with simpler feature sets."
)

st.markdown("### Model Algorithm Comparison")
st.markdown(
    """
    <div class="wn-card">
        <p>
            The final model experiment compared <strong>57 combinations</strong>: three feature sets
            (A, B and C) crossed with 19 model settings, including linear models, probabilistic text models,
            tree-based models, KNN and SVD-based variants. The table below lists the top 10 combinations ranked by
            Micro F1.
        </p>
        <span class="wn-pill">3 feature sets</span>
        <span class="wn-pill">19 model settings</span>
        <span class="wn-pill">57 combinations</span>
        <span class="wn-pill">Same AU/NZ split</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(
    [
        {"Rank": 1, "Feature set": "C: Optimized expanded features", "Model": "Linear SVM", "Micro F1": 0.730539, "Macro F1": 0.638981, "Micro precision": 0.715475, "Micro recall": 0.746251, "Hamming loss": 0.089907, "Exact match accuracy": 0.492653},
        {"Rank": 2, "Feature set": "C: Optimized expanded features", "Model": "Ridge Classifier", "Micro F1": 0.728975, "Macro F1": 0.641300, "Micro precision": 0.693761, "Micro recall": 0.767956, "Hamming loss": 0.093259, "Exact match accuracy": 0.470998},
        {"Rank": 3, "Feature set": "C: Optimized expanded features", "Model": "SGD SVM", "Micro F1": 0.725819, "Macro F1": 0.635676, "Micro precision": 0.701251, "Micro recall": 0.752170, "Hamming loss": 0.092807, "Exact match accuracy": 0.475638},
        {"Rank": 4, "Feature set": "C: Optimized expanded features", "Model": "Logistic Regression", "Micro F1": 0.699117, "Macro F1": 0.602364, "Micro precision": 0.623032, "Micro recall": 0.796369, "Hamming loss": 0.111949, "Exact match accuracy": 0.399845},
        {"Rank": 5, "Feature set": "C: Optimized expanded features", "Model": "Extra Trees", "Micro F1": 0.698189, "Macro F1": 0.551482, "Micro precision": 0.864298, "Micro recall": 0.585635, "Hamming loss": 0.082689, "Exact match accuracy": 0.522042},
        {"Rank": 6, "Feature set": "C: Optimized expanded features", "Model": "Calibrated Linear SVM", "Micro F1": 0.697202, "Macro F1": 0.585786, "Micro precision": 0.814015, "Micro recall": 0.609708, "Hamming loss": 0.086491, "Exact match accuracy": 0.511988},
        {"Rank": 7, "Feature set": "B: Ingredient + category", "Model": "Ridge Classifier", "Micro F1": 0.675227, "Macro F1": 0.598868, "Micro precision": 0.647357, "Micro recall": 0.705604, "Hamming loss": 0.110853, "Exact match accuracy": 0.385924},
        {"Rank": 8, "Feature set": "A: Ingredient list only", "Model": "Ridge Classifier", "Micro F1": 0.672855, "Macro F1": 0.595720, "Micro precision": 0.646995, "Micro recall": 0.700868, "Hamming loss": 0.111304, "Exact match accuracy": 0.386698},
        {"Rank": 9, "Feature set": "C: Optimized expanded features", "Model": "Random Forest", "Micro F1": 0.669779, "Macro F1": 0.486041, "Micro precision": 0.887370, "Micro recall": 0.537885, "Hamming loss": 0.086620, "Exact match accuracy": 0.513534},
        {"Rank": 10, "Feature set": "B: Ingredient + category", "Model": "Linear SVM", "Micro F1": 0.665370, "Macro F1": 0.590819, "Micro precision": 0.656178, "Micro recall": 0.674822, "Hamming loss": 0.110853, "Exact match accuracy": 0.383604},
    ],
    use_container_width=True,
    hide_index=True,
)

st.info(
    "The selected final App2.0 model is Feature Set C + Linear SVM. It achieved the highest Micro F1 among all 57 combinations and provided a strong balance across Macro F1, precision, recall, Hamming loss and exact match accuracy."
)

st.markdown(
    """
    <div class="wn-card">
        <p>
            <strong>Feature Set C + Linear SVM</strong> was selected as the final App2.0 model configuration. Although
            Extra Trees achieved the highest exact match accuracy, its recall was substantially lower. Linear SVM was
            therefore more suitable for undeclared allergen risk screening, where missing a potential risk is more
            serious than over-flagging.
        </p>
        <span class="wn-pill">Final feature set: C</span>
        <span class="wn-pill">Final model: Linear SVM</span>
        <span class="wn-pill">Best overall Micro F1</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Metric Meaning")
st.dataframe(
    [
        {"Metric": "Micro F1", "Meaning": "Overall multi-label F1 score across all labels.", "Why it matters": "Summarises general prediction performance."},
        {"Metric": "Macro F1", "Meaning": "Average F1 score treating each allergen label more equally.", "Why it matters": "Highlights performance on less common labels."},
        {"Metric": "Micro recall", "Meaning": "How many true allergen-risk labels the model finds overall.", "Why it matters": "Useful when missing a risk is more serious than over-flagging."},
        {"Metric": "Hamming loss", "Meaning": "Fraction of incorrect label decisions.", "Why it matters": "Lower is better for multi-label prediction."},
        {"Metric": "Exact match accuracy", "Meaning": "Percentage of products where the full predicted label set exactly matches the true set.", "Why it matters": "Strict but easy to interpret."},
    ],
    use_container_width=True,
    hide_index=True,
)

st.markdown('<div class="section-title">3. User-Facing Function Upgrades</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="wn-card">
        <h3>3.1 OCR and translation review</h3>
        <p>
            App2.0 improves the image-input workflow by separating OCR output into reviewable fields. Users can check
            the original OCR text, extracted ingredient list, translated ingredient list and translated allergen statement
            before the assessment is run.
        </p>
        <span class="wn-pill">Tesseract</span>
        <span class="wn-pill">EasyOCR</span>
        <span class="wn-pill">Field-level review</span>
        <span class="wn-pill">Translation check</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wn-card">
        <h3>3.2 Relative risk score output</h3>
        <p>
            App2.0 no longer presents the Linear SVM score as a true probability. The model output is shown as a
            relative risk score from 0 to 100, which indicates how strongly the label text resembles training examples
            associated with each allergen-risk label.
        </p>
        <p>
            The score is used for ranking and decision support, not as a regulatory threshold or a direct probability
            that the allergen is present.
        </p>
        <span class="wn-pill">Relative score /100</span>
        <span class="wn-pill">Model ranking</span>
        <span class="wn-pill">Not probability</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wn-card">
        <h3>3.3 Integrated cross-contact review</h3>
        <p>
            App2.0 combines the model ranking with derivative ingredient evidence. A top-ranked model signal is treated
            as Low / monitor when no supporting ingredient evidence is found, while derivative evidence such as gelatine,
            caseinate, soy lecithin or sulphite terms can raise an item for supplier verification.
        </p>
        <span class="wn-pill">Model rank</span>
        <span class="wn-pill">Derivative evidence</span>
        <span class="wn-pill">Low / monitor</span>
        <span class="wn-pill">Medium review</span>
        <span class="wn-pill">High review</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wn-card">
        <h3>3.4 Clearer decision-support wording</h3>
        <p>
            The final output has been renamed to <strong>Integrated Cross-Contact Allergen Risk Review</strong>. This
            makes the result easier to interpret because the page explains whether the signal comes from the model,
            derivative ingredient evidence, or both.
        </p>
        <span class="wn-pill">Evidence shown</span>
        <span class="wn-pill">Reason shown</span>
        <span class="wn-pill">Supplier verification</span>
    </div>
    """,
    unsafe_allow_html=True,
)

