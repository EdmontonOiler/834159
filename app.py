# ==================================================
# Allergen Risk Assessment App
# Text Input + Image OCR Input + International OCR
# Added: Unexpected Allergen Risk Detection
# ==================================================

import os
import re
import shutil
import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from deep_translator import GoogleTranslator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# --------------------------------------------------
# Tesseract path
# Compatible with local Windows and deployed Linux/cloud
# --------------------------------------------------
WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT
else:
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# --------------------------------------------------
# OCR language / country options
# --------------------------------------------------
LANG_OPTIONS = {
    "English": "eng",
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Japanese": "jpn",
    "Korean": "kor",
    "French": "fra",
    "German": "deu",
    "Spanish": "spa",
    "Italian": "ita",
    "Portuguese": "por",
    "Dutch": "nld",
    "Russian": "rus",
    "Thai": "tha"
}

COUNTRY_OPTIONS = {
    "Australia / USA / UK": "eng",
    "China": "chi_sim",
    "Taiwan / Hong Kong": "chi_tra",
    "Japan": "jpn",
    "Korea": "kor",
    "France": "fra",
    "Germany": "deu",
    "Spain": "spa",
    "Italy": "ita",
    "Portugal / Brazil": "por",
    "Netherlands": "nld",
    "Russia": "rus",
    "Thailand": "tha"
}

# --------------------------------------------------
# Load dataset and train model
# --------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_excel("wws data.xlsx")
    df.columns = df.columns.str.strip()

    X_text = df["Ingredient list"].astype(str)

    Y = df[[
        "maycontain_milk",
        "maycontain_egg",
        "maycontain_soy",
        "maycontain_peanut",
        "maycontain_tree_nuts",
        "maycontain_sesame",
        "maycontain_wheat",
        "maycontain_gluten",
        "maycontain_fish",
        "maycontain_shellfish",
        "maycontain_lupin",
        "maycontain_sulphites"
    ]]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(X_text)

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    model.fit(X, Y)

    return model, vectorizer, Y.columns


model, vectorizer, label_names = load_model()

# --------------------------------------------------
# Allergen name formatting
# --------------------------------------------------
def format_allergen_name(name):
    return name.replace("_", " ").title()

# --------------------------------------------------
# Rule-based allergen detection
# Multilingual keywords -> English allergen keys
# --------------------------------------------------
allergen_dict = {
    "milk": [
        "milk", "whey", "casein", "caseinate", "lactose", "butter", "cream", "cheese", "yogurt",
        "milk solids", "乳", "牛奶", "奶", "奶粉", "乳清", "酪蛋白",
        "ミルク", "乳成分", "牛乳", "우유", "유청", "카제인",
        "lait", "milch", "leche", "latte"
    ],
    "egg": [
        "egg", "albumin", "ovalbumin", "egg white", "egg yolk",
        "蛋", "鸡蛋", "雞蛋", "蛋白", "蛋黄",
        "卵", "卵白", "卵黄", "달걀", "계란",
        "oeuf", "ei", "huevo", "uovo"
    ],
    "soy": [
        "soy", "soya", "soybean", "soy lecithin", "tofu",
        "大豆", "黄豆", "黃豆", "大豆卵磷脂", "豆腐",
        "大豆レシチン", "大豆たんぱく", "대두", "콩", "대두 레시틴",
        "soja"
    ],
    "peanut": [
        "peanut", "groundnut",
        "花生", "落花生", "ピーナッツ", "땅콩",
        "arachide", "cacahuete", "amendoim"
    ],
    "tree_nuts": [
        "almond", "cashew", "walnut", "hazelnut", "macadamia", "pecan",
        "pistachio", "pine nut", "brazil nut",
        "杏仁", "腰果", "核桃", "榛子", "夏威夷果", "开心果", "開心果", "松子", "巴西坚果", "巴西堅果",
        "アーモンド", "カシューナッツ", "くるみ", "ヘーゼルナッツ", "マカダミア", "ピスタチオ",
        "아몬드", "캐슈넛", "호두", "헤이즐넛", "마카다미아", "피스타치오"
    ],
    "sesame": [
        "sesame", "tahini",
        "芝麻", "ごま", "ゴマ", "참깨",
        "sésame", "sesamo"
    ],
    "wheat": [
        "wheat", "flour",
        "小麦", "小麥", "面粉", "麵粉",
        "小麦粉", "밀", "밀가루",
        "blé", "trigo", "frumento"
    ],
    "gluten": [
        "gluten", "barley", "rye", "oats",
        "麸质", "麩質", "大麦", "大麥", "黑麦", "黑麥", "燕麦", "燕麥",
        "グルテン", "大麦", "ライ麦", "オーツ麦",
        "글루텐", "보리", "호밀", "귀리"
    ],
    "fish": [
        "fish", "salmon", "tuna", "cod", "anchovy", "sardine",
        "鱼", "魚", "三文鱼", "三文魚", "金枪鱼", "金槍魚", "鳕鱼", "鱈魚",
        "サーモン", "ツナ", "タラ", "アンチョビ", "イワシ",
        "생선", "연어", "참치", "대구", "멸치", "정어리"
    ],
    "shellfish": [
        "shrimp", "prawn", "crab", "lobster", "mussel", "clam", "oyster", "scallop",
        "虾", "蝦", "蟹", "龙虾", "龍蝦", "贻贝", "貽貝", "蛤", "牡蛎", "牡蠣", "扇贝", "扇貝",
        "えび", "カニ", "ロブスター", "ムール貝", "あさり", "かき", "ホタテ",
        "새우", "게", "바닷가재", "홍합", "조개", "굴", "가리비"
    ],
    "lupin": [
        "lupin", "羽扇豆", "ルピン", "루핀"
    ],
    "sulphites": [
        "sulphite", "sulfite", "sulphites", "sulfites",
        "亚硫酸盐", "亞硫酸鹽", "二氧化硫",
        "亜硫酸塩", "아황산염"
    ]
}

# --------------------------------------------------
# Unexpected allergen risk ingredients
# These are risk indicators, not confirmed allergens
# --------------------------------------------------
unexpected_source_dict = {
    "milk": [
        "lactate", "lactic acid", "casein", "caseinate", "whey",
        "lactose", "beverage whitener", "non-dairy creamer",
        "whitener", "brine", "lactoperoxidase"
    ],
    "egg": [
        "albumin", "albumen", "lysozyme", "glaze", "mayonnaise"
    ],
    "soy": [
        "lecithin", "soy lecithin", "tocopherols", "isoflavones",
        "hydrolysed vegetable protein", "hydrolyzed vegetable protein", "hvp",
        "textured vegetable protein"
    ],
    "wheat/gluten": [
        "malt", "malt extract", "maltodextrin", "dextrin", "dextrose",
        "glucose", "glucose syrup", "starch", "modified starch",
        "cornflour", "corn starch", "amylase", "vinegar",
        "ethanol", "yeast extract", "breadcrumbs"
    ],
    "fish": [
        "gelatine", "gelatin", "isinglass", "chitosan",
        "omega 3", "omega 6", "xanthophylls"
    ],
    "peanut/tree nut/sesame": [
        "vegetable oil", "oil", "fat", "fats", "cold pressed oil",
        "expeller pressed oil", "fatty acids", "mono-diglycerides",
        "mono and diglycerides", "glycerine", "glycerin", "oleoresins",
        "tahini"
    ],
    "general risk": [
        "flavour", "flavor", "flavours", "flavors",
        "colour", "color", "colours", "colors",
        "emulsifier", "emulsifiers", "stabiliser", "stabilizer",
        "stabilisers", "stabilizers", "thickener", "thickeners",
        "enzyme", "enzymes", "processing aid", "processing aids",
        "seasoning", "seasoning premix", "seasoning pre-mix",
        "spice extract", "herb extract"
    ],
    "sulphites": [
        "sulphite", "sulfite", "sulphites", "sulfites",
        "sulphur dioxide", "bisulphite", "bisulfite",
        "metabisulphite", "metabisulfite"
    ]
}

def detect_allergens(text):
    text = str(text).lower()
    detected = []

    for allergen, keywords in allergen_dict.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text):
                detected.append(allergen)
                break

    return detected

def detect_unexpected_risks(text, detected_allergens=None):
    text = str(text).lower()
    risks = []

    if detected_allergens is None:
        detected_allergens = []

    category_to_allergen_keys = {
        "milk": ["milk"],
        "egg": ["egg"],
        "soy": ["soy"],
        "wheat/gluten": ["wheat", "gluten"],
        "fish": ["fish"],
        "peanut/tree nut/sesame": ["peanut", "tree_nuts", "sesame"],
        "general risk": [],
        "sulphites": ["sulphites"]
    }

    for category, keywords in unexpected_source_dict.items():
        related_allergens = category_to_allergen_keys.get(category, [])

        if related_allergens and any(a in detected_allergens for a in related_allergens):
            continue

        found_keywords = []
        for kw in keywords:
            if kw.lower() in text:
                found_keywords.append(kw)

        if found_keywords:
            risks.append({
                "possible_allergen_group": category,
                "matched_terms": found_keywords
            })

    return risks

# --------------------------------------------------
# OCR image preprocessing
# --------------------------------------------------
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda x: 0 if x < 150 else 255, "1")
    image = image.convert("L")
    return image

def extract_text_from_image(image, lang="eng"):
    processed = preprocess_image(image)

    try:
        text = pytesseract.image_to_string(
            processed,
            lang=lang,
            config="--psm 6"
        )
        return text
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed on this environment. "
            "If you deployed this app, make sure packages.txt includes tesseract-ocr."
        )
    except pytesseract.TesseractError as e:
        raise RuntimeError(f"Tesseract OCR error: {e}")
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")

def clean_ocr_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_ingredient_section(text):
    patterns = [
        r"ingredients?\s*[:\-]\s*(.*)",
        r"ingredienti\s*[:\-]\s*(.*)",
        r"ingrédients?\s*[:\-]\s*(.*)",
        r"zutaten\s*[:\-]\s*(.*)",
        r"ingredientes?\s*[:\-]\s*(.*)",
        r"成分\s*[:：\-]\s*(.*)",
        r"配料\s*[:：\-]\s*(.*)",
        r"原料\s*[:：\-]\s*(.*)",
        r"原材料名\s*[:：\-]\s*(.*)",
        r"원재료명\s*[:：\-]\s*(.*)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return text.strip()

# --------------------------------------------------
# Translation
# --------------------------------------------------
def translate_to_english(text):
    text = str(text).strip()
    if not text:
        return ""

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation failed, using original text. Error: {e}")
        return text

# --------------------------------------------------
# ML prediction
# --------------------------------------------------
def predict_may_contain(text):
    X_new = vectorizer.transform([text])
    proba = model.predict_proba(X_new)

    results = []

    for i, label in enumerate(label_names):
        prob = float(proba[0][i])

        if prob > 0.5:
            name = label.replace("maycontain_", "")
            name = format_allergen_name(name)
            results.append((name, round(prob, 2)))

    return results

# --------------------------------------------------
# Shared analysis function
# --------------------------------------------------
def run_analysis(ingredient_text):
    ingredient_text = ingredient_text.strip()

    detected = detect_allergens(ingredient_text)
    predicted = predict_may_contain(ingredient_text)
    unexpected_risks = detect_unexpected_risks(ingredient_text, detected)

    st.subheader("Results")

    st.write("### Detected Allergens")
    if detected:
        detected_english = [format_allergen_name(a) for a in detected]
        st.success(", ".join(detected_english))
    else:
        st.write("None")

    st.write("### Predicted May Contain")
    if predicted:
        df_result = pd.DataFrame(
            predicted,
            columns=["Allergen", "Probability"]
        )
        st.dataframe(df_result, use_container_width=True)
    else:
        st.write("None")

    st.write("### Unexpected Allergen Risks")
    if unexpected_risks:
        rows = []
        for item in unexpected_risks:
            rows.append({
                "Possible Allergen Group": item["possible_allergen_group"],
                "Matched Terms": ", ".join(item["matched_terms"])
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.warning("These are risk indicators only. The ingredient source should be confirmed with supplier specifications.")
    else:
        st.write("None")

    if detected:
        st.info("Rule-based detection identified allergens explicitly listed in the ingredients.")
    if predicted:
        st.warning("The model also predicts possible 'may contain' risks based on learned ingredient patterns.")

# --------------------------------------------------
# Compliance checking
# --------------------------------------------------
def check_compliance(ingredient_text, statement_text):
    detected = detect_allergens(ingredient_text)
    unexpected_risks = detect_unexpected_risks(ingredient_text, detected)

    if pd.isna(statement_text):
        statement_text = ""

    statement_text = str(statement_text).lower()
    declared = []

    for allergen in allergen_dict.keys():
        allergen_name_for_check = allergen.replace("_", " ")
        if allergen_name_for_check in statement_text:
            declared.append(allergen)

    missing = [a for a in detected if a not in declared]
    compliant = len(missing) == 0

    may_contain_raw = predict_may_contain(ingredient_text)

    may_contain = [
        (a, p) for (a, p) in may_contain_raw
        if a.lower().replace(" ", "_") not in detected
    ]

    return {
        "detected_allergens": detected,
        "declared_allergens": declared,
        "missing_allergens": missing,
        "may_contain": may_contain,
        "unexpected_risks": unexpected_risks,
        "compliant": compliant
    }

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Allergen Risk Assessment App", layout="wide")

st.title("Allergen Risk Assessment App")
st.markdown("### Text input and image-based ingredient extraction")

tab1, tab2, tab3 = st.tabs([
    "Text Input",
    "Image OCR Input",
    "International Label OCR"
])

# --------------------------------------------------
# Tab 1: Manual text input
# --------------------------------------------------
with tab1:
    st.write("Enter the ingredient list directly.")

    ingredient_text_manual = st.text_area(
        "Ingredient List",
        placeholder="e.g. Sugar, Milk Solids, Cocoa Butter, Soy Lecithin, Wheat Flour",
        height=180,
        key="manual_text"
    )

    statement_text = st.text_area(
        "Allergen Statement (optional)",
        placeholder="e.g. May contain milk and soy",
        height=100
    )

    if st.button("Analyze Text", key="analyze_text"):
        result = check_compliance(ingredient_text_manual, statement_text)

        detected_display = [format_allergen_name(a) for a in result["detected_allergens"]]
        declared_display = [format_allergen_name(a) for a in result["declared_allergens"]]
        missing_display = [format_allergen_name(a) for a in result["missing_allergens"]]
        may_contain_display = [f"{a} ({p})" for a, p in result["may_contain"]]

        st.subheader("Results")

        st.success(f"Detected: {', '.join(detected_display) if detected_display else 'None'}")
        st.info(f"Declared: {', '.join(declared_display) if declared_display else 'None'}")

        if result["missing_allergens"]:
            st.error(f"Missing allergens: {', '.join(missing_display)}")
        else:
            st.success("No missing allergens")

        st.warning(f"May contain: {', '.join(may_contain_display) if may_contain_display else 'None'}")

        st.write("### Unexpected Allergen Risks")
        if result["unexpected_risks"]:
            rows = []
            for item in result["unexpected_risks"]:
                rows.append({
                    "Possible Allergen Group": item["possible_allergen_group"],
                    "Matched Terms": ", ".join(item["matched_terms"])
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.warning("These terms may indicate hidden allergen sources and should be checked with supplier specifications.")
        else:
            st.write("None")

        if result["compliant"]:
            st.success("Product is compliant")
        else:
            st.error("Product is NOT compliant")

# --------------------------------------------------
# Tab 2: Image OCR input
# --------------------------------------------------
with tab2:
    st.write("Upload a food label image. The app will extract text automatically, and you can edit it before analysis.")

    uploaded_file = st.file_uploader(
        "Upload food label image",
        type=["png", "jpg", "jpeg"],
        key="ocr_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Extract Text from Image", key="extract_ocr"):
            try:
                raw_text = extract_text_from_image(image, lang="eng")
                st.write("OCR raw text:", raw_text)

                cleaned_text = clean_ocr_text(raw_text)
                extracted_text = extract_ingredient_section(cleaned_text)
                st.session_state["ocr_text"] = extracted_text
            except Exception as e:
                st.error(str(e))

    ocr_text_value = st.text_area(
        "Extracted / Editable Ingredient Text",
        value=st.session_state.get("ocr_text", ""),
        height=180,
        key="editable_ocr_text"
    )

    if st.button("Analyze OCR Text", key="analyze_ocr"):
        if ocr_text_value.strip() == "":
            st.warning("Please extract or enter ingredient text first.")
        else:
            run_analysis(ocr_text_value)

# --------------------------------------------------
# Tab 3: International label OCR
# --------------------------------------------------
with tab3:
    st.write("Upload an international food label image. Select the country or language for OCR recognition.")

    ocr_mode_tab3 = st.radio(
        "Recognition mode",
        ["By Country", "By Language"],
        horizontal=True,
        key="tab3_ocr_mode"
    )

    selected_lang_tab3 = "eng"

    if ocr_mode_tab3 == "By Country":
        selected_country_tab3 = st.selectbox(
            "Select country / region",
            list(COUNTRY_OPTIONS.keys()),
            key="tab3_country"
        )
        selected_lang_tab3 = COUNTRY_OPTIONS[selected_country_tab3]

    elif ocr_mode_tab3 == "By Language":
        selected_language_tab3 = st.selectbox(
            "Select language",
            list(LANG_OPTIONS.keys()),
            key="tab3_language"
        )
        selected_lang_tab3 = LANG_OPTIONS[selected_language_tab3]

    uploaded_file_tab3 = st.file_uploader(
        "Upload international food label image",
        type=["png", "jpg", "jpeg"],
        key="ocr_upload_tab3"
    )

    if "editable_ocr_text_tab3" not in st.session_state:
        st.session_state["editable_ocr_text_tab3"] = ""

    if "translated_ocr_text_tab3" not in st.session_state:
        st.session_state["translated_ocr_text_tab3"] = ""

    if uploaded_file_tab3 is not None:
        image_tab3 = Image.open(uploaded_file_tab3)
        st.image(image_tab3, caption="Uploaded International Label", use_container_width=True)

        if st.button("Extract International Text", key="extract_ocr_tab3"):
            try:
                raw_text_tab3 = extract_text_from_image(image_tab3, lang=selected_lang_tab3)
                st.write("OCR raw text:", raw_text_tab3)

                cleaned_text_tab3 = clean_ocr_text(raw_text_tab3)
                extracted_text_tab3 = extract_ingredient_section(cleaned_text_tab3)
                translated_text_tab3 = translate_to_english(extracted_text_tab3)

                st.session_state["editable_ocr_text_tab3"] = extracted_text_tab3
                st.session_state["translated_ocr_text_tab3"] = translated_text_tab3

            except Exception as e:
                st.error(str(e))

    ocr_text_value_tab3 = st.text_area(
        "Extracted / Editable International Ingredient Text",
        height=180,
        key="editable_ocr_text_tab3"
    )

    translated_text_value_tab3 = st.text_area(
        "Translated English Ingredient Text",
        height=180,
        key="translated_ocr_text_tab3"
    )

    if st.button("Analyze International OCR Text", key="analyze_ocr_tab3"):
        if translated_text_value_tab3.strip() == "":
            st.warning("Please extract or enter ingredient text first.")
        else:
            run_analysis(translated_text_value_tab3)