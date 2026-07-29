# ==================================================
# Allergen Risk Assessment App 2.0
# Text Input + Image OCR Input + International OCR
# Added: Unexpected Allergen Risk Detection
# ==================================================

import os
import re
import shutil
import sqlite3
import math
from difflib import get_close_matches

import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from deep_translator import GoogleTranslator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRITION_DB_PATH = os.path.join(APP_DIR, "module1_data", "nutrition_ingredients.sqlite")
MODULE3_DATA_PATH = os.path.join(APP_DIR, "module3_data", "undeclared_allergen_training_au_nz.csv")

NUTRIENT_FIELDS = [
    ("Energy", "energy_kj", "energy_kj_per_100g", "kJ", 0),
    ("Protein", "protein_g", "protein_g_per_100g", "g", 1),
    ("Fat, total", "fat_total_g", "fat_total_g_per_100g", "g", 1),
    ("Saturated fat", "fat_saturated_g", "fat_saturated_g_per_100g", "g", 1),
    ("Carbohydrate", "carbohydrate_g", "carbohydrate_g_per_100g", "g", 1),
    ("Sugars", "sugars_g", "sugars_g_per_100g", "g", 1),
    ("Sodium", "sodium_mg", "sodium_mg_per_100g", "mg", 0),
]

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
    """Train the app model.

    App 2.0 prefers the AU/NZ Open Food Facts training set prepared for
    Module 3. If that file is not installed yet, the original WWS workbook is
    used as a fallback so the app remains runnable.
    """
    if os.path.exists(MODULE3_DATA_PATH):
        df = pd.read_csv(MODULE3_DATA_PATH)
        df.columns = df.columns.str.strip()

        X_text = df["ingredient_list"].fillna("").astype(str)

        crustacean = df["risk_crustacean"].astype(int) if "risk_crustacean" in df.columns else 0
        mollusc = df["risk_mollusc"].astype(int) if "risk_mollusc" in df.columns else 0
        df["risk_shellfish"] = (crustacean | mollusc).astype(int)

        Y = df[[
            "risk_milk",
            "risk_egg",
            "risk_soy",
            "risk_peanut",
            "risk_tree_nuts",
            "risk_sesame",
            "risk_wheat",
            "risk_gluten",
            "risk_fish",
            "risk_shellfish",
            "risk_lupin",
            "risk_sulphites"
        ]].astype(int)
    else:
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
        LinearSVC(class_weight="balanced", random_state=42)
    )

    model.fit(X, Y)

    training_decision_scores = model.decision_function(X)
    threshold_rows = []

    for i, label in enumerate(Y.columns):
        label_scores = [
            1.0 / (1.0 + math.exp(-max(min(float(raw), 20), -20)))
            for raw in training_decision_scores[:, i]
        ]
        threshold_rows.append({
            "label": str(label),
            "medium_threshold": float(pd.Series(label_scores).quantile(0.75)),
            "high_threshold": float(pd.Series(label_scores).quantile(0.90)),
        })

    risk_thresholds = {
        row["label"]: {
            "medium": row["medium_threshold"],
            "high": row["high_threshold"],
        }
        for row in threshold_rows
    }

    return model, vectorizer, Y.columns, risk_thresholds


model, vectorizer, label_names, risk_thresholds = load_model()

# --------------------------------------------------
# Allergen name formatting
# --------------------------------------------------
def format_allergen_name(name):
    name = str(name).replace("maycontain_", "").replace("risk_", "")
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


def detect_unexpected_risks(text, detected_allergens=None, declared_allergens=None):
    text = str(text).lower()
    risks = []

    if detected_allergens is None:
        detected_allergens = []
    if declared_allergens is None:
        declared_allergens = []

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

    high_signal_terms = {
        "casein", "caseinate", "whey", "lactose", "albumin", "albumen",
        "lysozyme", "soy lecithin", "hydrolysed vegetable protein",
        "hydrolyzed vegetable protein", "hvp", "textured vegetable protein",
        "malt", "malt extract", "breadcrumbs", "isinglass", "chitosan",
        "tahini", "sulphur dioxide", "bisulphite", "bisulfite",
        "metabisulphite", "metabisulfite"
    }

    medium_signal_terms = {
        "maltodextrin", "dextrin", "dextrose", "glucose", "glucose syrup",
        "starch", "modified starch", "gelatine", "gelatin", "tocopherols",
        "vegetable oil", "cold pressed oil", "expeller pressed oil",
        "mono-diglycerides", "mono and diglycerides", "glycerine", "glycerin",
        "flavour", "flavor", "flavours", "flavors", "enzyme", "enzymes",
        "processing aid", "processing aids"
    }

    reason_by_category = {
        "milk": "Matched terms may indicate milk-derived ingredients or dairy processing aids.",
        "egg": "Matched terms may indicate egg-derived proteins or egg-based processing aids.",
        "soy": "Matched terms may indicate soy-derived additives or protein ingredients.",
        "wheat/gluten": "Matched terms may be cereal-derived and should be checked for wheat or gluten source.",
        "fish": "Matched terms may indicate fish-derived clarifying agents, gelatine or marine ingredients.",
        "peanut/tree nut/sesame": "Matched terms may come from plant oils, nut/seed derivatives or shared source materials.",
        "sulphites": "Matched terms may indicate sulphite preservatives requiring declaration above regulatory thresholds.",
        "general risk": "Generic ingredients can hide variable supplier sources and should be verified."
    }

    for category, keywords in unexpected_source_dict.items():
        if category == "general risk":
            continue

        related_allergens = category_to_allergen_keys.get(category, [])
        already_detected = bool(related_allergens and any(a in detected_allergens for a in related_allergens))
        already_declared = bool(related_allergens and any(a in declared_allergens for a in related_allergens))

        found_keywords = []
        for kw in keywords:
            if kw.lower() in text:
                found_keywords.append(kw)

        if not found_keywords:
            continue

        if already_detected or already_declared:
            risk_level = "Declared / review optional"
            risk_color = "#64748b"
            recommended_action = "Allergen source appears already detected or declared; keep supplier evidence for records."
        elif any(term.lower() in high_signal_terms for term in found_keywords):
            risk_level = "High"
            risk_color = "#dc2626"
            recommended_action = "Check supplier specification and confirm whether allergen declaration or PAL assessment is required."
        elif any(term.lower() in medium_signal_terms for term in found_keywords):
            risk_level = "Medium"
            risk_color = "#f59e0b"
            recommended_action = "Verify raw material source with supplier and review allergen management documents."
        else:
            risk_level = "Low"
            risk_color = "#16a34a"
            recommended_action = "Record as low-priority review unless supplier source or product category suggests higher risk."

        risks.append({
            "possible_allergen_group": category,
            "matched_terms": found_keywords,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "reason": reason_by_category.get(category, "Matched terms may indicate an indirect allergen source."),
            "recommended_action": recommended_action,
            "already_declared_or_detected": already_detected or already_declared,
        })

    risk_order = {"High": 0, "Medium": 1, "Low": 2, "Declared / review optional": 3}
    risks.sort(key=lambda item: risk_order.get(item["risk_level"], 9))
    return risks

# --------------------------------------------------
# OCR image preprocessing and recognition
# App 2.0 uses EasyOCR as the primary visual text recognition engine.
# Tesseract is kept only as a fallback if EasyOCR is not installed.
# --------------------------------------------------
EASYOCR_LANG_MAP = {
    "eng": "en",
    "chi_sim": "ch_sim",
    "chi_tra": "ch_tra",
    "jpn": "ja",
    "kor": "ko",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "rus": "ru",
    "tha": "th",
}


def get_easyocr_languages(lang="eng"):
    mapped = EASYOCR_LANG_MAP.get(lang, "en")
    if mapped == "en":
        return ["en"]
    return [mapped, "en"]


@st.cache_resource
def load_easyocr_reader(languages):
    import easyocr

    return easyocr.Reader(list(languages), gpu=False)


def preprocess_image_for_easyocr(image, max_width=1400):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, max(1, int(image.height * ratio)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def preprocess_image_for_tesseract(image):
    image = ImageOps.exif_transpose(image)
    image = ImageOps.grayscale(image)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda x: 0 if x < 150 else 255, "1")
    image = image.convert("L")
    return image


def extract_text_with_easyocr(image, lang="eng"):
    import numpy as np

    languages = tuple(get_easyocr_languages(lang))
    reader = load_easyocr_reader(languages)
    processed = preprocess_image_for_easyocr(image)
    results = reader.readtext(np.array(processed), detail=1, paragraph=False)

    sorted_results = sorted(
        results,
        key=lambda item: (
            min(point[1] for point in item[0]) if item and item[0] else 0,
            min(point[0] for point in item[0]) if item and item[0] else 0,
        ),
    )

    lines = []
    confidences = []
    for item in sorted_results:
        if len(item) >= 2:
            lines.append(str(item[1]).strip())
        if len(item) >= 3:
            try:
                confidences.append(float(item[2]))
            except Exception:
                pass

    text = "\n".join(line for line in lines if line)
    if not text.strip():
        raise RuntimeError("EasyOCR did not detect readable text from this image.")

    st.session_state["ocr_engine"] = "EasyOCR"
    st.session_state["ocr_average_confidence"] = (
        sum(confidences) / len(confidences) if confidences else None
    )
    st.session_state["ocr_detected_text_blocks"] = len(lines)
    return text


def extract_text_with_tesseract_fallback(image, lang="eng"):
    processed = preprocess_image_for_tesseract(image)
    text = pytesseract.image_to_string(
        processed,
        lang=lang,
        config="--psm 6"
    )
    st.session_state["ocr_engine"] = "Tesseract fallback"
    st.session_state["ocr_average_confidence"] = None
    st.session_state["ocr_detected_text_blocks"] = None
    return text


def extract_text_from_image(image, lang="eng", engine="Accurate OCR (EasyOCR)"):
    if engine == "Fast OCR (Tesseract)":
        try:
            return extract_text_with_tesseract_fallback(image, lang=lang)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract is not installed or cannot be found. Use Accurate OCR (EasyOCR), or install Tesseract."
            )
        except pytesseract.TesseractError as tesseract_error:
            raise RuntimeError(f"Tesseract OCR failed: {tesseract_error}")

    try:
        return extract_text_with_easyocr(image, lang=lang)
    except ModuleNotFoundError:
        st.warning("EasyOCR is not installed yet. Tesseract fallback is being used for this run.")
        return extract_text_with_tesseract_fallback(image, lang=lang)
    except Exception as easyocr_error:
        try:
            st.warning(f"EasyOCR failed, using Tesseract fallback. Error: {easyocr_error}")
            return extract_text_with_tesseract_fallback(image, lang=lang)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Neither EasyOCR nor Tesseract is available. Install EasyOCR with: pip install easyocr"
            )
        except pytesseract.TesseractError as tesseract_error:
            raise RuntimeError(
                f"EasyOCR failed ({easyocr_error}) and Tesseract also failed ({tesseract_error})."
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"EasyOCR failed ({easyocr_error}) and fallback OCR failed ({fallback_error})."
            )


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
            extracted = match.group(1).strip()
            extracted = re.split(r"\bcontains\b", extracted, flags=re.IGNORECASE)[0].strip()
            return extracted

    return re.split(r"\bcontains\b", text.strip(), flags=re.IGNORECASE)[0].strip()


def extract_allergen_statement(text):
    patterns = [
        r"\bcontains\b\s+[^.:\n]+",
        r"\bcontains\b\s*[:\-]\s*[^.\n]+"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return ""

# --------------------------------------------------
# Translation
# --------------------------------------------------

def assess_ocr_quality(raw_text, ingredient_text, statement_text):
    raw_text = str(raw_text or "").strip()
    ingredient_text = str(ingredient_text or "").strip()
    statement_text = str(statement_text or "").strip()

    issues = []
    if len(raw_text) < 30:
        issues.append("OCR text is very short")
    if len(ingredient_text) < 20:
        issues.append("Ingredient section may be incomplete")
    if not re.search(r"ingredient|ingredients|contains|may contain", raw_text, flags=re.IGNORECASE):
        issues.append("No clear ingredient/allergen keyword detected")
    if len(raw_text) > 0:
        readable_ratio = sum(ch.isalnum() or ch.isspace() or ch in ",.;:()[]/-" for ch in raw_text) / len(raw_text)
        if readable_ratio < 0.72:
            issues.append("OCR text contains many unclear characters")
    if not statement_text:
        issues.append("No allergen statement detected")

    if len(issues) <= 1 and len(ingredient_text) >= 30:
        return "Good OCR quality", "#16a34a", issues or ["Ingredient text appears usable"]
    if len(issues) <= 3:
        return "Needs review", "#f59e0b", issues
    return "Low confidence", "#dc2626", issues


def suggest_ocr_language_from_text(text):
    text = str(text or "")
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese text detected"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese text detected"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean text detected"
    return "No non-English script detected"


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
    decision_scores = model.decision_function(X_new)

    results = []

    for i, label in enumerate(label_names):
        raw_score = float(decision_scores[0][i])
        bounded_score = max(min(raw_score, 20), -20)
        risk_score = 1.0 / (1.0 + math.exp(-bounded_score))
        thresholds = risk_thresholds.get(str(label), {"medium": 0.75, "high": 0.90})
        name = format_allergen_name(label)
        results.append((name, risk_score, thresholds["medium"], thresholds["high"]))

    return results

# --------------------------------------------------
# Compliance checking
# --------------------------------------------------
def check_compliance(ingredient_text, statement_text):
    detected = detect_allergens(ingredient_text)

    if pd.isna(statement_text):
        statement_text = ""

    statement_text = str(statement_text).lower()
    declared = []

    for allergen in allergen_dict.keys():
        allergen_name_for_check = allergen.replace("_", " ")
        if allergen_name_for_check in statement_text:
            declared.append(allergen)

    unexpected_risks = detect_unexpected_risks(ingredient_text, detected, declared)
    missing = [a for a in detected if a not in declared]
    compliant = len(missing) == 0

    may_contain_raw = predict_may_contain(ingredient_text)

    may_contain = [
        (a, p, medium_threshold, high_threshold)
        for (a, p, medium_threshold, high_threshold) in may_contain_raw
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
# Module 1: Nutrition information generation
# --------------------------------------------------
def normalise_ingredient_name(value):
    return " ".join(str(value).strip().lower().split())

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

def render_nutrition_module():
    st.markdown('<div id="nutrition-section"></div>', unsafe_allow_html=True)
    st.markdown("## Nutrition Information Generation")
    st.write(
        "Generate a prototype Nutrition Information Panel from formulation data. "
        "Ingredient names are standardised using the local nutrition database."
    )

    if not os.path.exists(NUTRITION_DB_PATH):
        st.error("Nutrition database not found. Please check module1_data/nutrition_ingredients.sqlite.")
        return

    with st.container(border=True):
        st.write("### Product information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            product_name = st.text_input("Product name", "Butter biscuit prototype")
        with c2:
            final_weight = st.number_input("Final product weight (g)", min_value=1.0, value=530.0, step=10.0)
        with c3:
            serving_size = st.number_input("Serving size (g)", min_value=1.0, value=40.0, step=1.0)
        with c4:
            package_weight = st.number_input("Package weight (g)", min_value=1.0, value=530.0, step=10.0)

        st.write("### Recipe formulation")
        default_recipe = pd.DataFrame(
            [
                {"ingredient_name": "plain flour", "amount_g": 200.0},
                {"ingredient_name": "caster sugar", "amount_g": 100.0},
                {"ingredient_name": "butter", "amount_g": 80.0},
                {"ingredient_name": "egg", "amount_g": 50.0},
                {"ingredient_name": "milk", "amount_g": 100.0},
            ]
        )

        recipe_df = st.data_editor(
            default_recipe,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ingredient_name": st.column_config.TextColumn("Ingredient name"),
                "amount_g": st.column_config.NumberColumn("Amount (g)", min_value=0.0, step=1.0),
            },
            key="nutrition_recipe_editor",
        )




        if st.button("Generate Nutrition Information Panel", key="generate_nip"):
            result = calculate_nutrition_panel(recipe_df, final_weight, serving_size, package_weight)

            st.session_state["nutrition_result"] = result
            st.session_state["nutrition_product_name"] = product_name
            st.session_state["nutrition_serving_size"] = serving_size

    if "nutrition_result" in st.session_state:
        result = st.session_state["nutrition_result"]
        st.markdown("### Calculation status")

        if result["unmatched_count"] == 0:
            st.success(result["status"])
        else:
            st.warning(result["status"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Servings per package", f"{result['servings_per_package']:.1f}")
        m2.metric("Serving size", f"{st.session_state['nutrition_serving_size']:.0f} g")
        m3.metric("Unmatched ingredients", result["unmatched_count"])

        st.markdown("### Nutrition Information Panel")
        st.dataframe(result["nip"], use_container_width=True, hide_index=True)

        st.markdown("### Ingredient matching")
        st.dataframe(result["matches"], use_container_width=True, hide_index=True)

        with st.expander("View ingredient nutrient contributions"):
            st.dataframe(result["contributions"], use_container_width=True, hide_index=True)

        st.download_button(
            "Download NIP CSV",
            result["nip"].to_csv(index=False).encode("utf-8"),
            file_name="nutrition_information_panel.csv",
            mime="text/csv",
        )

# ==================================================
# Streamlit UI
# ==================================================
st.set_page_config(page_title="Allergen Risk Assessment App 2.0", layout="wide")

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
    
                    .risk-note .info-icon {
                        width: 22px !important;
                        height: 22px !important;
                        min-width: 22px !important;
                        flex: 0 0 22px !important;
                        display: inline-flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        box-sizing: border-box !important;
                        line-height: 1 !important;
                        font-family: Arial, sans-serif !important;
                        font-weight: 800 !important;
                        margin-top: 2px !important;
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
    [data-testid="stSidebarNav"] ul {
        gap: 6px;
    }

    /* Short sidebar titles */
    [data-testid="stSidebarNav"] ul li:first-child a span {
        font-size: 0;
    }
    [data-testid="stSidebarNav"] ul li:first-child a span::after {
        content: "App2.0";
        font-size: 16px;
        font-weight: 700;
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
    .sidebar-card {
        background: linear-gradient(180deg, #10264f 0%, #1d4ed8 100%);
        color: white;
        padding: 18px 16px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 12px 30px rgba(30, 64, 175, 0.20);
    }
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 18px;
    }
    .sidebar-section-label {
        font-size: 13px;
        opacity: 0.8;
        margin-top: 12px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .sidebar-item {
        display: block;
        padding: 11px 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        background: rgba(255,255,255,0.07);
        font-size: 14px;
        color: white !important;
        text-decoration: none !important;
        transition: all 0.25s ease;
    }
    .sidebar-item:hover {
        background: rgba(255,255,255,0.18);
        transform: translateX(3px);
        color: white !important;
        text-decoration: none !important;
    }
    .sidebar-item:visited {
        color: white !important;
        text-decoration: none !important;
    }
    .sidebar-item:active {
        color: white !important;
        text-decoration: none !important;
    }
    .sidebar-item.active {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.22);
        font-weight: 700;
    }
    .kpi-card {
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 110px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi-title {
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .kpi-body {
        font-size: 15px;
        line-height: 1.35;
        font-weight: 600;
    }
    .animated-bar-track {
        width: 100%;
        background: #eceff3;
        border-radius: 999px;
        height: 22px;
        overflow: hidden;
        margin-top: 4px;
    }
    .animated-bar-fill {
        width: 0;
        height: 22px;
        border-radius: 999px;
        color: white;
        font-size: 12px;
        font-weight: 700;
        line-height: 22px;
        text-align: right;
        padding-right: 8px;
        white-space: nowrap;
        animation: growBar 1.1s ease-out forwards;
        transform-origin: left center;
    }
    @keyframes growBar {
        from { width: 0; }
        to { width: var(--target-width); }
    }
    .result-panel {
        background: white;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .risk-card {
        margin-bottom: 12px;
        padding: 12px;
        border-radius: 10px;
        background: #fff7e6;
        border: 1px solid #f6d58f;
    }
    .risk-title {
        font-weight: 700;
        color: #9a6700;
    }
    .risk-subtext {
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= HEADER =================

# CSS
st.markdown("""
<style>
.app-banner {
    padding: 26px 28px;
    border-radius: 16px;
    background: linear-gradient(135deg, #2f80ed, #1e3a8a);
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(30,58,138,0.25);
}

.app-title-row {
    display: flex;
    align-items: center;
    gap: 16px;
}

.app-icon {
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: rgba(255,255,255,0.15);
    display: flex;
    align-items: center;
    justify-content: center;
}

.app-title {
    font-size: 40px;
    font-weight: 900;
}

.app-subtitle {
    font-size: 20px;
    margin-top: 6px;
    opacity: 0.9;
}

.app-caption {
    font-size: 14px;
    opacity: 0.85;
    margin-top: 4px;
}

.app-divider {
    height: 3px;
    margin-top: 16px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ffffff, rgba(255,255,255,0.4), transparent);
}
</style>
""", unsafe_allow_html=True)

# HTML
st.markdown("""
<div class="app-banner">

<div class="app-title-row">
<div class="app-icon">
<svg width="26" height="26" viewBox="0 0 24 24" fill="none"
     stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M4 20h16"></path>
<path d="M6 16V8"></path>
<path d="M12 16V4"></path>
<path d="M18 16v-6"></path>
</svg>
</div>

<div>
<div class="app-title">Allergen Risk Assessment App 2.0</div>
<div class="app-subtitle">Allergen Risk Analyzer</div>
<div class="app-caption">Analyze ingredient lists using manual input or image OCR.</div>
</div>
</div>

<div class="app-divider"></div>

</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
"""<div class="sidebar-card">
<div class="sidebar-title">Allergen Risk<br>Assessment App</div>
<div class="sidebar-section-label">Sections</div>

<a href="#input-section" class="sidebar-item active">Input</a>
<a href="#review-section" class="sidebar-item">Review & Edit</a>
<a href="#results-section" class="sidebar-item">Results Dashboard</a>

</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("**Quick guide**")
    st.caption(
        "1. Upload or enter ingredients\n\n"
        "2. Review and edit extracted text\n\n"
        "3. Run risk assessment"
    )

# Session state initialization
if "ingredient_text" not in st.session_state:
    st.session_state["ingredient_text"] = ""

if "statement_text" not in st.session_state:
    st.session_state["statement_text"] = ""

if "original_ocr_text" not in st.session_state:
    st.session_state["original_ocr_text"] = ""

# --------------------------------------------------
# Input section
# --------------------------------------------------
st.markdown('<div id="input-section"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .input-title-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 22px;
        padding-bottom: 18px;
        border-bottom: 1px solid #e5e7eb;
    }

    .input-title-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #eaf3ff, #dbeafe);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(47,128,237,0.18);
    }

    .input-title-text {
        font-size: 34px;
        font-weight: 900;
        color: #0f172a;
        letter-spacing: 0.2px;
    }

    .input-step-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px 22px;
        margin-top: 18px;
        margin-bottom: 20px;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(15,23,42,0.04);
        transition: all 0.25s ease;
    }

    .input-step-card:hover {
        box-shadow: 0 8px 22px rgba(15,23,42,0.08);
        transform: translateY(-1px);
    }

    .step-heading {
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 21px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 18px;
        letter-spacing: 0.2px;
    }

    .step-heading::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #dbeafe, transparent);
        margin-left: 10px;
    }

    .step-badge {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, #2f80ed, #1d4ed8);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 17px;
        box-shadow: 0 4px 10px rgba(47,128,237,0.35);
        transition: all 0.25s ease;
        flex-shrink: 0;
    }

    .step-badge:hover {
        transform: scale(1.08);
        box-shadow: 0 6px 14px rgba(47,128,237,0.45);
    }

    [data-testid="stFileUploader"] {
        border: 2px dashed #3b82f6 !important;
        border-radius: 14px !important;
        padding: 18px !important;
        background: #f8fbff !important;
        transition: all 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        background: #eef5ff !important;
        border-color: #2563eb !important;
    }

    [data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        padding: 12px 22px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.45) !important;
        background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):

    st.markdown(
        """
        <div class="input-title-row">
            <div class="input-title-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                     stroke="#2f80ed" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 20h9"></path>
                    <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"></path>
                </svg>
            </div>
            <div class="input-title-text">Input</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Choose input method",
        ["Image OCR", "Manual Text"],
        horizontal=True,
        key="input_method"
    )

    if input_method == "Image OCR":

        st.markdown('<div id="ocr-section"></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="input-step-card">
                <div class="step-heading">
                    <div class="step-badge">1</div>
                    <div>Step 1 · Upload image and choose OCR settings</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        engine_col, lang_mode_col, lang_select_col = st.columns([1.25, 1, 1.7], gap="large")

        with engine_col:
            selected_ocr_engine = st.radio(
                "OCR engine",
                ["Fast OCR (Tesseract)", "Accurate OCR (EasyOCR)"],
                index=0,
                horizontal=False,
                key="selected_ocr_engine",
                help="Fast OCR is quicker for clear English labels. Accurate OCR is better for complex or multilingual labels.",
            )

        with lang_mode_col:
            ocr_mode = st.radio(
                "Recognition mode",
                ["By Country", "By Language"],
                horizontal=False,
                key="ocr_mode"
            )

        selected_lang = "eng"

        with lang_select_col:
            if ocr_mode == "By Country":
                selected_country = st.selectbox(
                    "Select country / region",
                    list(COUNTRY_OPTIONS.keys()),
                    key="country_select"
                )
                selected_lang = COUNTRY_OPTIONS[selected_country]
            else:
                selected_language = st.selectbox(
                    "Select language",
                    list(LANG_OPTIONS.keys()),
                    key="language_select"
                )
                selected_lang = LANG_OPTIONS[selected_language]

        uploaded_file = st.file_uploader(
            "Upload food label image",
            type=["png", "jpg", "jpeg"],
            key="ocr_upload"
        )

        if uploaded_file is not None:
            current_upload_signature = (
                uploaded_file.name,
                uploaded_file.size,
                uploaded_file.type,
                selected_ocr_engine,
                selected_lang,
            )

            if st.session_state.get("last_ocr_upload_signature") != current_upload_signature:
                for key in [
                    "ocr_raw_text",
                    "ocr_cleaned_text",
                    "original_ocr_text",
                    "ocr_original_statement",
                    "ocr_translated_ingredient_text",
                    "ocr_translated_statement_text",
                    "ocr_quality_label",
                    "ocr_quality_color",
                    "ocr_quality_issues",
                    "ocr_language_code",
                    "ocr_language_suggestion",
                    "ocr_engine",
                    "ocr_average_confidence",
                    "ocr_detected_text_blocks",
                    "translation_applied",
                    "ingredient_text",
                    "statement_text",
                ]:
                    st.session_state.pop(key, None)

                st.session_state["last_ocr_upload_signature"] = current_upload_signature
                st.rerun()

            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", width=300)

            if st.button("Extract Text", key="extract_ocr"):
                try:
                    raw_text = extract_text_from_image(
                        image,
                        lang=selected_lang,
                        engine=selected_ocr_engine,
                    )
                    cleaned_text = clean_ocr_text(raw_text)
                    extracted_text = extract_ingredient_section(cleaned_text)
                    extracted_statement = extract_allergen_statement(cleaned_text)

                    translated_ingredients = extracted_text
                    translated_statement = extracted_statement
                    translation_applied = selected_lang != "eng"

                    if translation_applied:
                        translated_ingredients = translate_to_english(extracted_text)
                        if extracted_statement:
                            translated_statement = translate_to_english(extracted_statement)

                    quality_label, quality_color, quality_issues = assess_ocr_quality(
                        cleaned_text,
                        extracted_text,
                        extracted_statement,
                    )

                    st.session_state["ocr_raw_text"] = raw_text
                    st.session_state["ocr_cleaned_text"] = cleaned_text
                    st.session_state["original_ocr_text"] = extracted_text
                    st.session_state["ocr_original_statement"] = extracted_statement
                    st.session_state["ocr_translated_ingredient_text"] = translated_ingredients
                    st.session_state["ocr_translated_statement_text"] = translated_statement
                    st.session_state["ocr_quality_label"] = quality_label
                    st.session_state["ocr_quality_color"] = quality_color
                    st.session_state["ocr_quality_issues"] = quality_issues
                    st.session_state["ocr_language_code"] = selected_lang
                    st.session_state["ocr_language_suggestion"] = suggest_ocr_language_from_text(cleaned_text)
                    st.session_state["translation_applied"] = translation_applied
                    st.session_state["input_source"] = "Image OCR"

                    st.session_state["ingredient_text"] = translated_ingredients
                    st.session_state["statement_text"] = translated_statement
                    st.rerun()

                except Exception as e:
                    st.error(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get("ocr_cleaned_text"):
            quality_label = st.session_state.get("ocr_quality_label", "Needs review")
            quality_color = st.session_state.get("ocr_quality_color", "#f59e0b")
            quality_issues = st.session_state.get("ocr_quality_issues", [])
            translation_status = "Applied" if st.session_state.get("translation_applied") else "Not required"

            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;margin:18px 0;background:#ffffff;">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px;">
                        <div style="font-size:18px;font-weight:900;color:#0f172a;">OCR and Translation Review</div>
                        <div style="background:{quality_color};color:white;padding:6px 12px;border-radius:999px;font-weight:800;font-size:13px;">
                            {quality_label}
                        </div>
                    </div>
                    <div style="color:#475569;font-size:14px;">
                        OCR engine: {st.session_state.get("ocr_engine", "EasyOCR")} · OCR language: {st.session_state.get("ocr_language_code", "eng")} · Confidence: {st.session_state.get("ocr_average_confidence", "N/A")} · Translation: {translation_status} · {st.session_state.get("ocr_language_suggestion", "")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if quality_issues:
                st.caption("Quality notes: " + "; ".join(quality_issues))

            original_col, translated_col = st.columns(2, gap="large")

            with original_col:
                st.text_area(
                    "Original OCR text",
                    value=st.session_state.get("ocr_cleaned_text", ""),
                    height=120,
                    disabled=True,
                    key="ocr_cleaned_text_display",
                )
                st.text_area(
                    "Extracted ingredient list",
                    value=st.session_state.get("original_ocr_text", ""),
                    height=120,
                    disabled=True,
                    key="ocr_original_ingredients_display",
                )
                st.text_area(
                    "Extracted allergen statement",
                    value=st.session_state.get("ocr_original_statement", ""),
                    height=80,
                    disabled=True,
                    key="ocr_original_statement_display",
                )

            with translated_col:
                st.text_area(
                    "Translated ingredient list",
                    value=st.session_state.get("ocr_translated_ingredient_text", ""),
                    height=120,
                    disabled=True,
                    key="ocr_translated_ingredients_display",
                )
                st.text_area(
                    "Translated allergen statement",
                    value=st.session_state.get("ocr_translated_statement_text", ""),
                    height=80,
                    disabled=True,
                    key="ocr_translated_statement_display",
                )

        st.markdown(
            """
            <div class="input-step-card">
                <div class="step-heading">
                    <div class="step-badge">2</div>
                    <div>Step 2 · Review extracted text</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.info("Enter ingredient list manually.")

        st.markdown(
            """
            <div class="input-step-card">
                <div class="step-heading">
                    <div class="step-badge">1</div>
                    <div>Step 1 · Enter ingredient information</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div id="review-section"></div>', unsafe_allow_html=True)

    ingredient_text = st.text_area(
        "Ingredient List",
        height=140,
        key="ingredient_text"
    )

    statement_text = st.text_area(
        "Allergen Statement (optional)",
        placeholder="e.g. Contains milk and soy",
        height=90,
        key="statement_text"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="input-step-card">
            <div class="step-heading">
                <div class="step-badge">3</div>
                <div>Step 3 · Run assessment</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    run_analysis = st.button("▶ Run Risk Assessment", key="analyze")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div id="analysis-section"></div>', unsafe_allow_html=True)
st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Analysis
# --------------------------------------------------
if run_analysis:

    if ingredient_text.strip() == "":
        st.warning("Please enter or extract ingredient text first.")

    else:
        result = check_compliance(ingredient_text, statement_text)

        detected_display = [format_allergen_name(a) for a in result["detected_allergens"]]
        declared_display = [format_allergen_name(a) for a in result["declared_allergens"]]
        missing_display = [format_allergen_name(a) for a in result["missing_allergens"]]

        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="
                    width:36px;
                    height:36px;
                    border-radius:10px;
                    background:#eef5ff;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                ">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                         stroke="#2f80ed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="4" y1="20" x2="4" y2="10"></line>
                        <line x1="10" y1="20" x2="10" y2="4"></line>
                        <line x1="16" y1="20" x2="16" y2="14"></line>
                        <line x1="22" y1="20" x2="22" y2="8"></line>
                    </svg>
                </div>
                <div style="font-size:30px;font-weight:800;">
                    Results
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        main_left, main_right = st.columns([1, 1], gap="large")

        # =========================
        # LEFT SIDE
        # =========================
        with main_left:
            with st.container(border=True):

                st.markdown(
                    """
                    <style>
                    .kpi-card {
                        border-radius: 10px;
                        padding: 20px 22px;
                        height: 160px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                        border: 1px solid rgba(0,0,0,0.06);
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                        margin-bottom: 18px;
                    }

                    .kpi-header {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    }

                    .kpi-icon {
                        width: 34px;
                        height: 34px;
                        border-radius: 50%;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 18px;
                        font-weight: 800;
                    }

                    .kpi-title {
                        font-size: 16px;
                        font-weight: 800;
                        color: #0f172a;
                    }

                    .kpi-main {
                        font-size: 26px;
                        font-weight: 900;
                    }

                    .kpi-body {
                        font-size: 16px;
                        font-weight: 700;
                        color: #0f172a;
                    }

                    .risk-bg {
                        background: #fff1f2;
                        border-color: #ffd0d5;
                    }

                    .success-bg {
                        background: #e6f4ea;
                        border-color: #b7e1cd;
                    }

                    .safe-bg {
                        background: #eef5ff;
                        border-color: #d4e5ff;
                    }

                    .red-icon { background: #ef4444; }
                    .green-icon { background: #22c55e; }
                    .blue-icon { background: #2f80ed; }

                    .red-text { color: #d60000; }
                    .green-text { color: #16a34a; }

                    .derivative-title-row {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        margin: 20px 0;
                    }

                    .warning-icon {
                        width: 36px;
                        height: 36px;
                        border-radius: 10px;
                        background: #fff3cd;
                        color: #b26a00;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 22px;
                        font-weight: 800;
                    }

                    .derivative-title {
                        font-size: 20px;
                        font-weight: 800;
                        color: #1f2a44;
                    }

                    .risk-card {
                        margin-bottom: 20px;
                        padding: 20px;
                        border-radius: 12px;
                        background: #fff8e8;
                        border: 1px solid #ffd98a;
                    }

                    .risk-title {
                        font-size: 18px;
                        font-weight: 800;
                        color: #9a5b00;
                    }

                    .risk-subtext {
                        font-size: 14px;
                        color: #64748b;
                    }

                    .risk-note {
                        margin-top: 20px;
                        padding: 20px;
                        border-radius: 12px;
                        background: #f3f6fb;
                        color: #64748b;
                        display: flex;
                        gap: 10px;
                    }

                    .info-icon {
                        width: 22px;
                        height: 22px;
                        border-radius: 50%;
                        border: 2px solid #7c8da6;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 14px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                top_left, top_right = st.columns(2)
                bottom_left, bottom_right = st.columns(2)

                is_ok = result["compliant"]

                with top_left:
                    st.markdown(
                        f"""
                        <div class="kpi-card {'success-bg' if is_ok else 'risk-bg'}">
                            <div class="kpi-header">
                                <div class="kpi-icon {'green-icon' if is_ok else 'red-icon'}">
                                    {"✓" if is_ok else "!"}
                                </div>
                                <div class="kpi-title">Compliance</div>
                            </div>
                            <div>
                                <div class="kpi-main {'green-text' if is_ok else 'red-text'}">
                                    {"COMPLIANT" if is_ok else "NOT COMPLIANT"}
                                </div>
                                <div class="kpi-body">
                                    {"All detected allergens are declared" if is_ok else "Label mismatch detected"}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with top_right:
                    st.markdown(
                        f"""
                        <div class="kpi-card {'success-bg' if not missing_display else 'risk-bg'}">
                            <div class="kpi-header">
                                <div class="kpi-icon {'green-icon' if not missing_display else 'red-icon'}">
                                    {"✓" if not missing_display else "!"}
                                </div>
                                <div class="kpi-title">Missing Allergens</div>
                            </div>
                            <div>
                                <div class="kpi-main {'green-text' if not missing_display else 'red-text'}">
                                    {len(missing_display)}
                                </div>
                                <div class="kpi-body">
                                    {', '.join(missing_display) if missing_display else 'None'}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with bottom_left:
                    st.markdown(
                        f"""
                        <div class="kpi-card safe-bg">
                            <div class="kpi-header">
                                <div class="kpi-icon blue-icon">✓</div>
                                <div class="kpi-title">Detected Allergens</div>
                            </div>
                            <div class="kpi-body">
                                {', '.join(detected_display) if detected_display else 'None'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with bottom_right:
                    st.markdown(
                        f"""
                        <div class="kpi-card safe-bg">
                            <div class="kpi-header">
                                <div class="kpi-icon blue-icon">▣</div>
                                <div class="kpi-title">Declared Allergens</div>
                            </div>
                            <div class="kpi-body">
                                {', '.join(declared_display) if declared_display else 'None'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    """
                    <div class="derivative-title-row">
                        <div class="warning-icon">!</div>
                        <div class="derivative-title">
                            Derivative Ingredient Risk Review
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if result["unexpected_risks"]:
                    risk_rows = []
                    high_count = sum(1 for item in result["unexpected_risks"] if item.get("risk_level") == "High")
                    medium_count = sum(1 for item in result["unexpected_risks"] if item.get("risk_level") == "Medium")

                    st.markdown(
                        f"""
                        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-bottom:14px;background:#ffffff;">
                            <div style="font-size:15px;color:#64748b;font-weight:700;">Derivative risk summary</div>
                            <div style="font-size:24px;font-weight:900;color:#0f172a;margin-top:4px;">
                                {len(result["unexpected_risks"])} group(s) detected
                            </div>
                            <div style="font-size:14px;color:#64748b;margin-top:4px;">
                                High: {high_count} · Medium: {medium_count} · Review supplier source before final label decision.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    for item in result["unexpected_risks"]:
                        group = item["possible_allergen_group"].title()
                        terms = ", ".join(item["matched_terms"])
                        risk_level = item.get("risk_level", "Medium")
                        risk_color = item.get("risk_color", "#f59e0b")
                        reason = item.get("reason", "")
                        action = item.get("recommended_action", "Verify ingredient source.")

                        risk_rows.append({
                            "Risk group": group,
                            "Matched terms": terms,
                            "Risk level": risk_level,
                            "Reason": reason,
                            "Recommended action": action,
                        })

                        st.markdown(
                            f"""
                            <div style="margin-bottom:14px;padding:16px 18px;border-radius:12px;background:#ffffff;border:1px solid #e5e7eb;">
                                <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px;">
                                    <div style="font-size:17px;font-weight:900;color:#0f172a;">{group}</div>
                                    <div style="background:{risk_color};color:white;padding:5px 10px;border-radius:999px;font-size:13px;font-weight:900;white-space:nowrap;">
                                        {risk_level}
                                    </div>
                                </div>
                                <div style="font-size:14px;color:#334155;margin-bottom:8px;">
                                    <strong>Matched terms:</strong> {terms}
                                </div>
                                <div style="font-size:14px;color:#64748b;margin-bottom:8px;">
                                    {reason}
                                </div>
                                <div style="font-size:14px;color:#0f172a;background:#f8fafc;border-radius:8px;padding:10px 12px;">
                                    <strong>Recommended action:</strong> {action}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with st.expander("View derivative risk summary table"):
                        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)
                else:
                    st.success("No derivative ingredient risk terms detected.")

                st.markdown(
                    """
                    <div class="risk-note">
                        <div class="info-icon">i</div>
                        <div>
                            This section is decision support only. It flags indirect or derivative ingredient terms that may require supplier verification, not confirmed undeclared allergens.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)

        # =========================
        # RIGHT SIDE
        # =========================
        with main_right:
            with st.container(border=True):

                st.markdown(
                    """
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                        <span style="width:24px;height:24px;border-radius:7px;background:#eef6ff;border:1px solid #d7e8ff;display:inline-flex;align-items:flex-end;justify-content:center;gap:2px;padding:4px;box-sizing:border-box;">
                            <span style="width:4px;height:9px;border-radius:3px;background:#60a5fa;display:block;"></span>
                            <span style="width:4px;height:14px;border-radius:3px;background:#34d399;display:block;"></span>
                            <span style="width:4px;height:18px;border-radius:3px;background:#a78bfa;display:block;"></span>
                        </span>
                        <div style="font-size:20px;font-weight:850;line-height:1.22;color:#1f2937;">
                            Integrated Cross-Contact Allergen Risk Review
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div style="margin-bottom:14px;padding:12px 14px;border-radius:10px;background:#f8fafc;border:1px solid #dbeafe;color:#334155;font-size:14px;line-height:1.55;">
                        Relative risk score is used as a model-ranking signal, not as a confirmed probability.
                        Final review priority combines the model rank with derivative ingredient evidence and declaration status.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                filtered = [(a, p, m, h) for (a, p, m, h) in result["may_contain"] if p >= 0.001]
                filtered = sorted(filtered, key=lambda x: x[1], reverse=True)

                evidence_group_map = {
                    "Milk": ["milk"],
                    "Egg": ["egg"],
                    "Soy": ["soy"],
                    "Wheat": ["wheat/gluten"],
                    "Gluten": ["wheat/gluten"],
                    "Fish": ["fish"],
                    "Peanut": ["peanut/tree nut/sesame"],
                    "Tree Nuts": ["peanut/tree nut/sesame"],
                    "Sesame": ["peanut/tree nut/sesame"],
                    "Sulphites": ["sulphites"],
                    "Shellfish": ["general risk"],
                    "Lupin": ["general risk"],
                }

                def evidence_for_allergen(allergen_name):
                    groups = evidence_group_map.get(allergen_name, [])
                    matches = []
                    for item in result.get("unexpected_risks", []):
                        if item.get("possible_allergen_group") in groups:
                            matches.append(item)
                    return matches

                def final_review_priority(rank, evidence_items):
                    if any(item.get("risk_level") == "High" for item in evidence_items):
                        return (
                            "High review",
                            "#dc2626",
                            "Supporting derivative evidence was detected, and supplier/allergen documentation should be reviewed before final label decision.",
                        )
                    if any(item.get("risk_level") == "Medium" for item in evidence_items):
                        return (
                            "Medium review",
                            "#f59e0b",
                            "Derivative ingredient evidence was detected. Review ingredient source or supplier documentation.",
                        )
                    if rank <= 3:
                        return (
                            "Low / monitor",
                            "#16a34a",
                            "This is a top-ranked model signal, but no supporting derivative evidence was found in the label text.",
                        )
                    return (
                        "Low review",
                        "#16a34a",
                        "No supporting derivative evidence was found. Keep standard label review.",
                    )

                if filtered:
                    priority_counts = {"High review": 0, "Medium review": 0, "Low / monitor": 0, "Low review": 0}
                    prepared_rows = []

                    for rank, (allergen, prob, medium_threshold, high_threshold) in enumerate(filtered, start=1):
                        evidence_items = evidence_for_allergen(allergen)
                        review_label, color, explanation = final_review_priority(rank, evidence_items)
                        priority_counts[review_label] = priority_counts.get(review_label, 0) + 1

                        evidence_text = "No supporting derivative evidence found."
                        if evidence_items:
                            evidence_text = "; ".join(
                                f"{item.get('possible_allergen_group')}: {', '.join(item.get('matched_terms', []))}"
                                for item in evidence_items
                            )

                        prepared_rows.append((rank, allergen, prob, review_label, color, explanation, evidence_text))

                    st.markdown(
                        f"""
                        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-bottom:14px;">
                            <div style="padding:10px 12px;border-radius:10px;background:#fef2f2;border:1px solid #fecaca;">
                                <div style="font-size:12px;color:#991b1b;font-weight:800;">High review</div>
                                <div style="font-size:24px;font-weight:900;color:#7f1d1d;">{priority_counts.get('High review', 0)}</div>
                            </div>
                            <div style="padding:10px 12px;border-radius:10px;background:#fffbeb;border:1px solid #fde68a;">
                                <div style="font-size:12px;color:#92400e;font-weight:800;">Medium review</div>
                                <div style="font-size:24px;font-weight:900;color:#78350f;">{priority_counts.get('Medium review', 0)}</div>
                            </div>
                            <div style="padding:10px 12px;border-radius:10px;background:#f0fdf4;border:1px solid #bbf7d0;">
                                <div style="font-size:12px;color:#166534;font-weight:800;">Low / monitor</div>
                                <div style="font-size:24px;font-weight:900;color:#14532d;">{priority_counts.get('Low / monitor', 0) + priority_counts.get('Low review', 0)}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    for rank, allergen, prob, review_label, color, explanation, evidence_text in prepared_rows:
                        percent = prob * 100
                        st.markdown(
                            f"""
                            <div style="margin-bottom:16px;padding:14px 16px;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;">
                                <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px;">
                                    <div>
                                        <span style="font-weight:900;color:#0f172a;">{allergen}</span>
                                        <span style="margin-left:8px;color:#64748b;font-size:13px;font-weight:700;">Model rank #{rank}</span>
                                    </div>
                                    <span style="background:{color};color:white;padding:5px 10px;border-radius:999px;font-size:13px;font-weight:900;white-space:nowrap;">
                                        {review_label}
                                    </span>
                                </div>
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;color:#64748b;font-size:13px;">
                                    <span>Relative risk score</span>
                                    <span>{percent:.1f}/100</span>
                                </div>
                                <div class="animated-bar-track">
                                    <div class="animated-bar-fill"
                                         style="--target-width:{percent:.1f}%; background:{color};">
                                        {percent:.1f}
                                    </div>
                                </div>
                                <div style="margin-top:10px;font-size:13px;color:#334155;line-height:1.55;">
                                    <strong>Evidence:</strong> {evidence_text}<br>
                                    <strong>Reason:</strong> {explanation}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:
                    st.write("None")
