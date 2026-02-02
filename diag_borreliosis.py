# -*- coding: utf-8 -*-
"""
LYRAE / RESOLVE — Streamlit predictor (CatBoost)
Single-file diag_borreliosis.py (ou app.py)

✅ Correction appliquée (erreur StreamlitAPIException sur session_state):
- PROBLÈME: faire `st.session_state["addr_num"] = st.text_input(..., key="addr_num")`
  -> Streamlit interdit d'écrire dans st.session_state["addr_num"] quand un widget utilise le même key.
- FIX: ne JAMAIS assigner directement à st.session_state[...] pour une clé de widget.
  -> On lit le retour du widget dans une variable (num = st.text_input(...))
  -> Et on laisse Streamlit gérer st.session_state automatiquement.
  -> On utilise st.session_state.setdefault(...) avant pour valeurs par défaut.

Le reste du code est inchangé + complet.
"""

import json
import io
import time
import requests
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from catboost import CatBoostClassifier, Pool

# Raster risk
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer


# ============================================================
# APP CONFIG
# ============================================================
APP_BRAND = "LYRAE"
APP_TITLE = "Aide au diagnostic de la borréliose de Lyme équine"
APP_SUBTITLE = "Analyse structurée basée sur les données cliniques, biologiques et contextuelles."
MODEL_DEFAULT = "equine_lyme_catboost.cbm"
META_DEFAULT  = "equine_lyme_catboost_meta.json"

REF_XLSX_URL = "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/jeu_fictif_lyme_equine_cas_parfaits.xlsx"
REF_XLSX_SHEET = 0
REF_XLSX_IGNORE = {"target", "y", "label"}

HERO_IMAGE_URL = "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/Lyrae.png"
MINI_LOGO_URL  = "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/minilyrae.png"

# Raster de risque (catégories 1/2/3)
RISK_RASTER_URL = "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/mean_R1_RF_prob_rep01_05_CATEG_3classes.tif"

# Recommandation : mettre une vraie adresse mail de contact pour le User-Agent
CONTACT_EMAIL = (
    st.secrets.get("contact_email", "contact@exemple.org")
    if hasattr(st, "secrets")
    else "contact@exemple.org"
)


# ============================================================
# VARIABLES (met à jour selon ton modèle si besoin)
# ============================================================
analysis_cols = [
    "piroplasmose_neg","ehrlichiose_neg","ehrlichiose_negatif","Bilan_sanguin_normal","NFS_normale",
    "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
    "SAA_normal","Fibrinogène_normal",
    "ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","WB_pos","PCR_sang_pos","SNAP_C6_pos","IFAT_pos",
    "PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos",
    "PCR_liquide_articulaire_pos","LCR_pleiocytose","LCR_proteines_augmentees",
    "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos",
    "CVID","Hypoglobulinemie"
]

RESULTS_ANALYSIS_COLS = [
    "ELISA_pos", "ELISA_OspA_pos", "ELISA_OspF_pos", "ELISA_p39",
    "WB_pos", "SNAP_C6_pos", "IFAT_pos",
    "PCR_sang_pos", "PCR_LCR_pos", "PCR_synoviale_pos", "PCR_liquide_articulaire_pos",
    "PCR_peau_pos", "PCR_humeur_aqueuse_pos", "PCR_tissu_nerveux_pos",
    "LCR_pleiocytose", "LCR_proteines_augmentees",
    "IHC_tissulaire_pos", "Coloration_argent_pos", "FISH_tissulaire_pos",
    "CVID", "Hypoglobulinemie",
]

st.set_page_config(page_title=f"{APP_BRAND} — {APP_TITLE}", layout="wide")


# ============================================================
# STYLE (premium + tabs pill + inputs luxe + suppressions UI)
# ============================================================
CSS = """
<style>
/* ------------------------------------------------------------
   Streamlit chrome
------------------------------------------------------------ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Optionnel: cache le header blanc "Streamlit" résiduel selon versions */
div[data-testid="stHeader"] { display: none !important; }

/* ------------------------------------------------------------
   Design tokens
------------------------------------------------------------ */
:root{
  --g900:#0e3b35;
  --g850:#124640;
  --g800:#154b43;

  --beige:#f4f2ed;
  --beige2:#efe9df;

  --ink:#1d2a2a;

  --accent:#b08b5a;
  --accent2:#d2b48c;

  --card: rgba(255,255,255,.78);
  --card-strong: rgba(255,255,255,.88);

  --shadow-soft: 0 8px 22px rgba(0,0,0,.10);
  --shadow-premium: 0 14px 34px rgba(0,0,0,.08);
  --shadow-focus: 0 0 0 4px rgba(176,139,90,.18);

  --radius: 18px;
  --radius-xl: 22px;
}

/* ------------------------------------------------------------
   Base layout
------------------------------------------------------------ */
.stApp{
  background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--beige) 60%, var(--beige2) 100%);
  color: var(--ink);
}

.block-container{
  padding-top: 0rem;
  padding-bottom: 2.2rem;
  max-width: 1100px;
}

/* Un peu plus d'air entre sections */
.block-container > div { gap: 14px; }

/* ------------------------------------------------------------
   Topbar
------------------------------------------------------------ */
.lyrae-topbar{
  position: sticky;
  top: 0;
  z-index: 999;
  background: linear-gradient(180deg, var(--g900) 0%, var(--g800) 100%);
  box-shadow: 0 4px 18px rgba(0,0,0,.18);
  padding: 14px 22px;
  margin: 0 -9999px;
  padding-left: calc(22px + 9999px);
  padding-right: calc(22px + 9999px);
}

.lyrae-topbar-inner{
  max-width: 1100px;
  margin: 0 auto;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 16px;
}

.lyrae-brand{
  display:flex;
  align-items:center;
  gap: 12px;
  color: #ffffff;
  font-weight: 900;
  letter-spacing: .6px;
  font-size: 20px;
}

.lyrae-logo{
  width: 38px;
  height: 38px;
  border-radius: 50%;
  overflow:hidden;
  box-shadow: 0 8px 18px rgba(0,0,0,.25);
  border: 1px solid rgba(255,255,255,.25);
  flex: 0 0 auto;
}
.lyrae-logo img{ width:100%; height:100%; display:block; }

/* ------------------------------------------------------------
   Tabs — pill premium
------------------------------------------------------------ */
div[data-testid="stTabs"]{
  margin-top: 6px !important;
}

/* Pastilles */
div[data-testid="stTabs"] button[role="tab"]{
  border-radius: 999px !important;
  padding: 10px 16px !important;
  margin-right: 10px !important;
  background: rgba(255,255,255,.65) !important;
  border: 1px solid rgba(14,59,53,.14) !important;
  box-shadow: 0 6px 16px rgba(0,0,0,.05) !important;
  color: rgba(14,59,53,.96) !important;
  font-weight: 880 !important;
  transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}

div[data-testid="stTabs"] button[role="tab"]:hover{
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(0,0,0,.08) !important;
  background: rgba(255,255,255,.78) !important;
}

/* Sélection */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  background: rgba(14,59,53,.10) !important;
  border: 1px solid rgba(14,59,53,.30) !important;
  box-shadow: 0 12px 26px rgba(0,0,0,.10) !important;
}

div[data-baseweb="tab-highlight"]{
  background-color: transparent !important; /* on n'utilise plus la barre */
}

/* ------------------------------------------------------------
   (Optionnel) Masquer la "barre/zone blanche vide" sous les tabs
   (varie selon version Streamlit -> deux sélecteurs)
------------------------------------------------------------ */
div[data-testid="stTabs"] + div > div:empty { display: none !important; }
div[data-testid="stTabs"] + div:has(> div:empty) { display: none !important; }

/* ------------------------------------------------------------
   Inputs — luxe + focus ring
------------------------------------------------------------ */

/* Labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label{
  color: var(--g900) !important;
  font-weight: 880 !important;
  letter-spacing: .2px;
}

/* Text & number wrapper */
div[data-testid="stTextInput"] > div > div,
div[data-testid="stNumberInput"] > div > div{
  border-radius: 14px !important;
  border: 1px solid rgba(14,59,53,.22) !important;
  background: rgba(255,255,255,.82) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.55);
}

/* Inner input */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
  padding: 12px 12px !important;
}

/* Focus ring */
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus{
  outline: none !important;
  box-shadow: var(--shadow-focus) !important;
  border: 1px solid rgba(176,139,90,.65) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] div[role="combobox"]{
  background: rgba(14,59,53,.90) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(14,59,53,.28) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,.06);
}
div[data-testid="stSelectbox"] div[role="combobox"] *{ color: #ffffff !important; }

div[role="listbox"]{
  background: rgba(14,59,53,.96) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.16) !important;
}
div[role="listbox"] *{ color: #ffffff !important; }

/* ------------------------------------------------------------
   Hero
------------------------------------------------------------ */
.lyrae-hero{ padding: 56px 0 24px 0; text-align: center; }
.lyrae-hero h1{ margin:0; font-size:42px; line-height:1.12; font-weight:900; color: var(--g900); letter-spacing: .2px; }
.lyrae-hero p{ margin:14px auto 0 auto; max-width:820px; font-size:17px; color:rgba(29,42,42,.64); }

.lyrae-illustration{
  margin: 30px auto 24px auto;
  border-radius: 22px;
  background: radial-gradient(900px 260px at 50% 30%, #ffffff 0%, #f7f4ee 45%, #f2efe8 100%);
  box-shadow: var(--shadow-soft);
  overflow:hidden;
  border: 1px solid rgba(0,0,0,.05);
}

/* ------------------------------------------------------------
   Titles / cards
------------------------------------------------------------ */
.lyrae-cta-wrap{ display:flex; align-items:center; justify-content:center; margin-top:18px; }
.lyrae-disclaimer{ margin-top:18px; color:rgba(29,42,42,.58); font-size:14px; }

.lyrae-page-title{ margin: 26px 0 6px 0; font-size: 30px; font-weight: 950; color: var(--g900); letter-spacing: .2px; }

.lyrae-card{
  background: var(--card);
  border: 1px solid rgba(14,59,53,.12);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 18px 18px 12px 18px;
}
.lyrae-card h3{ margin:0 0 10px 0; font-size:18px; font-weight:900; color: var(--g900); }

/* Premium card (pour "Identité" etc.) */
.lyrae-card--premium{
  background: var(--card-strong);
  border: 1px solid rgba(14,59,53,.12);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-premium);
  padding: 18px 18px 14px 18px;
}

/* Header premium */
.lyrae-card-header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 12px;
  margin-bottom: 12px;
}
.lyrae-card-title{
  display:flex;
  align-items:center;
  gap: 10px;
  font-size: 18px;
  font-weight: 950;
  color: var(--g900);
}
.lyrae-card-sub{
  margin-top: 4px;
  color: rgba(29,42,42,.62);
  font-size: 13px;
  font-weight: 650;
}
.lyrae-badge{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(14,59,53,.08);
  border: 1px solid rgba(14,59,53,.16);
  color: rgba(14,59,53,.92);
  font-weight: 900;
  font-size: 12px;
}

/* ------------------------------------------------------------
   Buttons
------------------------------------------------------------ */
.stButton > button, .stDownloadButton > button{
  border-radius: 14px !important;
  padding: 0.80rem 1.15rem !important;
  font-weight: 900 !important;
  border: 1px solid rgba(14,59,53,.22) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,.08);
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}

.stButton > button{
  background: linear-gradient(180deg, var(--accent2) 0%, var(--accent) 100%) !important;
  color: rgba(14,59,53,.98) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(0,0,0,.10);
  filter: brightness(1.02);
}

.stButton > button:active,
.stDownloadButton > button:active{
  transform: translateY(0px);
  box-shadow: 0 10px 22px rgba(0,0,0,.08);
}

/* ------------------------------------------------------------
   Result card
------------------------------------------------------------ */
.lyrae-result{
  border-radius: 20px;
  padding: 18px 18px;
  color: white;
  font-weight: 950;
  font-size: 22px;
  text-align: center;
  box-shadow: 0 12px 28px rgba(0,0,0,.12);
  border: 1px solid rgba(255,255,255,.28);
}
.lyrae-result small{
  display:block;
  margin-top:8px;
  font-size: 14px;
  font-weight: 750;
  opacity: .92;
}

.lyrae-scale{
  margin-top: 14px;
  border-radius: 16px;
  height: 16px;
  background: linear-gradient(90deg, #2e7d32 0%, #f9a825 45%, #ef6c00 70%, #c62828 100%);
  position: relative;
  box-shadow: inset 0 2px 8px rgba(0,0,0,.12);
}
.lyrae-marker{
  position: absolute;
  top: -6px;
  width: 10px;
  height: 28px;
  border-radius: 8px;
  background: rgba(255,255,255,.95);
  box-shadow: 0 6px 16px rgba(0,0,0,.18);
  transform: translateX(-50%);
}

/* Pills */
.lyrae-mini-pill{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(14,59,53,.08);
  border: 1px solid rgba(14,59,53,.14);
  color: rgba(14,59,53,.92);
  font-weight: 850;
  font-size: 12px;
}

/* ------------------------------------------------------------
   Expander + dataframe (petit polish)
------------------------------------------------------------ */
details{
  border-radius: 16px !important;
}

div[data-testid="stExpander"]{
  border-radius: 16px !important;
  border: 1px solid rgba(14,59,53,.10) !important;
  background: rgba(255,255,255,.55) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,.05) !important;
}

/* ------------------------------------------------------------
   Scrollbar (subtil)
------------------------------------------------------------ */
*::-webkit-scrollbar{ width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb{
  background: rgba(14,59,53,.25);
  border-radius: 999px;
  border: 3px solid rgba(255,255,255,.55);
}
*::-webkit-scrollbar-track{ background: rgba(255,255,255,.35); }

/* ------------------------------------------------------------
   FIX: barre blanche ovale sous les tabs (panel container)
------------------------------------------------------------ */

/* Le wrapper juste après la zone des tabs */
div[data-testid="stTabs"] + div{
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
  padding-top: 0 !important;
  margin-top: 0 !important;
}

/* Le premier bloc que Streamlit crée parfois (vide/arrondi) */
div[data-testid="stTabs"] + div > div:first-child{
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* Si Streamlit met une “carte” arrondie vide */
div[data-testid="stTabs"] + div > div:first-child:empty{
  display: none !important;
}

/* ------------------------------------------------------------
   Labels sur une seule ligne (no-wrap + ellipsis)
------------------------------------------------------------ */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label{
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  display: block !important;
  max-width: 100% !important;
}

/* ------------------------------------------------------------
   Widgets pleine largeur (encadrés même taille)
------------------------------------------------------------ */
div[data-testid="stTextInput"],
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"]{
  width: 100% !important;
}

div[data-testid="stTextInput"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stSelectbox"] > div{
  width: 100% !important;
}

/* Selectbox : le combobox prend toute la largeur */
div[data-testid="stSelectbox"] div[role="combobox"]{
  width: 100% !important;
}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



# ============================================================
# HELPERS (généraux)
# ============================================================
def normalize_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "_")
    return s


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    for k in ("feature_cols", "cat_cols", "factor_levels"):
        if k not in meta:
            raise ValueError(f"meta.json invalide: clé manquante '{k}'")
    return meta


@st.cache_resource
def load_model_and_meta(model_path_str: str, meta_path_str: str):
    model_path = Path(model_path_str)
    meta_path = Path(meta_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta introuvable: {meta_path}")

    meta = load_meta(meta_path)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    factor_levels = meta["factor_levels"]

    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    return model, meta, feature_cols, cat_cols, factor_levels, cat_idx


@st.cache_data(show_spinner=False)
def load_xlsx_columns(url: str, sheet=0) -> list[str]:
    try:
        headers = {"User-Agent": f"LYRAE-Streamlit/1.0 ({CONTACT_EMAIL})"}
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        bio = io.BytesIO(r.content)
        df = pd.read_excel(bio, sheet_name=sheet, engine="openpyxl")
        return [str(c).strip() for c in df.columns]
    except Exception as e:
        return [f"__ERROR__:{type(e).__name__}:{e}"]


def yn_to_num_if_needed(val, col_is_numeric: bool):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    if pd.isna(val):
        return val
    if not col_is_numeric:
        return val
    if isinstance(val, (int, float, np.number)) and not pd.isna(val):
        return float(val)
    s = str(val).strip().lower()
    if s in ("oui","yes","y","true","vrai","1"):
        return 1.0
    if s in ("non","no","n","false","faux","0"):
        return 0.0
    return val


def build_template(feature_cols):
    return pd.DataFrame([{c: pd.NA for c in feature_cols}])


def apply_inputs_to_template(X, inputs: dict):
    for k, v in inputs.items():
        if k in X.columns:
            X.at[0, k] = v
    return X


def fill_missing_code_like_R(X: pd.DataFrame, analysis_cols_set: set):
    miss_cols = [c for c in X.columns if c.endswith("_missing_code")]
    if not miss_cols:
        return X
    for mc in miss_cols:
        X.at[0, mc] = 0
    for mc in miss_cols:
        base = mc.replace("_missing_code", "")
        if base not in X.columns:
            continue
        if pd.isna(X.at[0, base]):
            X.at[0, mc] = 2 if base in analysis_cols_set else 1
    return X


def coerce_like_train_python(
    X: pd.DataFrame,
    feature_cols: list,
    cat_cols: list,
    factor_levels: dict
):
    """
    Rend X compatible CatBoost (Pool) :
    - Cat features : toujours string, jamais pd.NA dans les catégories
    - Numériques : float coerced
    """
    # --- Catégorielles : forcer en string + sentinel pour NA
    for c in cat_cols:
        if c not in X.columns:
            continue

        # On force en "string" pandas, puis on remplace les NA par un token
        s = X[c].astype("string")
        s = s.fillna("__MISSING__")

        # Optionnel : si meta fournit des niveaux, on peut "aligner" sans casser
        # (CatBoost accepte des strings hors niveaux, mais ça peut signaler une dérive)
        lv = factor_levels.get(c, None)
        if lv is not None and isinstance(lv, (list, tuple)) and len(lv) > 0:
            # On conserve la valeur telle quelle, mais on s'assure que les NA sont déjà gérés
            pass

        # CatBoost préfère object/str simples plutôt que Categorical pandas
        X[c] = s.astype(str)

    # --- Numériques : convertir Oui/Non -> 1/0, puis to_numeric
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for c in num_cols:
        if c not in X.columns:
            continue
        X[c] = X[c].apply(lambda v: yn_to_num_if_needed(v, col_is_numeric=True))
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X



def cat_from_p_like_R(p: float) -> str:
    if p < 0.25:
        return "Pas de Lyme ou informations insuffisantes"
    if p < 0.50:
        return "Lyme possible"
    if p < 0.75:
        return "Lyme probable"
    return "Lyme sûr"


def cat_color(cat: str) -> str:
    if cat.startswith("Pas de Lyme"):
        return "linear-gradient(180deg, #2e7d32 0%, #1b5e20 100%)"
    if cat == "Lyme possible":
        return "linear-gradient(180deg, #f9a825 0%, #f57f17 100%)"
    if cat == "Lyme probable":
        return "linear-gradient(180deg, #ef6c00 0%, #e65100 100%)"
    return "linear-gradient(180deg, #c62828 0%, #8e0000 100%)"


# ============================================================
# RISQUE AUTO via raster
# ============================================================
@st.cache_data(show_spinner=False)
def download_risk_raster(url: str) -> str:
    local_path = str(Path(__file__).with_name("mean_R1_RF_prob_rep01_05_CATEG_3classes.tif"))
    p = Path(local_path)
    if p.exists() and p.stat().st_size > 0:
        return local_path

    headers = {"User-Agent": f"LYRAE-Streamlit/1.0 ({CONTACT_EMAIL})"}
    r = requests.get(url, headers=headers, timeout=90)
    r.raise_for_status()
    p.write_bytes(r.content)
    return local_path


def _best_match_risk_label(target: str, levels: list[str]) -> str:
    if not levels:
        return target
    t = target.strip().lower()

    for lv in levels:
        if str(lv).strip().lower() == t:
            return str(lv)

    def pick(keyword):
        for lv in levels:
            if keyword in str(lv).strip().lower():
                return str(lv)
        return None

    if "faible" in t or "meconnu" in t or "méconnu" in t:
        return pick("faible") or pick("meconnu") or pick("méconnu") or target
    if "inter" in t:
        return pick("inter") or target
    if "fort" in t:
        return pick("fort") or target
    return target


def risk_class_from_geo(lat_wgs84: float, lon_wgs84: float, factor_levels: dict) -> str | None:
    try:
        tif_path = download_risk_raster(RISK_RASTER_URL)
        with rasterio.open(tif_path) as ds:
            ds_crs = ds.crs
            if ds_crs is None:
                return None

            transformer = Transformer.from_crs("EPSG:4326", ds_crs, always_xy=True)
            x, y = transformer.transform(lon_wgs84, lat_wgs84)

            if (x < ds.bounds.left) or (x > ds.bounds.right) or (y < ds.bounds.bottom) or (y > ds.bounds.top):
                raw_label = "faible ou méconnu"
            else:
                row, col = rowcol(ds.transform, x, y)
                if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                    raw_label = "faible ou méconnu"
                else:
                    v = ds.read(1, window=((row, row + 1), (col, col + 1)))
                    if v is None or v.size == 0:
                        raw_label = "faible ou méconnu"
                    else:
                        vv = v[0, 0]
                        try:
                            vv_int = int(vv)
                        except Exception:
                            vv_int = 1

                        if vv_int == 2:
                            raw_label = "intermédiaire"
                        elif vv_int == 3:
                            raw_label = "fort"
                        else:
                            raw_label = "faible ou méconnu"

        lv = factor_levels.get("Classe_de_risque", [])
        return _best_match_risk_label(raw_label, [str(x) for x in lv]) if lv else raw_label
    except Exception:
        return None


# ============================================================
# GEOCODE + MAP (Leaflet) — robuste FR (BAN -> Nominatim)
#   ✅ évite de "cacher" un échec (None) à cause du cache Streamlit
# ============================================================

def geocode_address(address: str):
    """
    Retourne {"lat":..., "lon":..., "display_name":..., "provider":...} ou None.
    Stratégie:
      1) BAN (France) -> très fiable
      2) Nominatim fallback
    """
    if not address or address.strip() == "":
        return None

    q = address.strip()

    # --- 1) BAN (Base Adresse Nationale) : fiable en France
    try:
        ban_url = "https://api-adresse.data.gouv.fr/search/"
        ban_params = {"q": q, "limit": 1}
        r = requests.get(
            ban_url,
            params=ban_params,
            timeout=12,
            headers={
                "User-Agent": f"LYRAE/1.0 ({CONTACT_EMAIL})",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.6",
            },
        )
        if r.status_code == 200:
            data = r.json()
            feats = data.get("features", [])
            if feats:
                coords = feats[0]["geometry"]["coordinates"]  # [lon, lat]
                props = feats[0].get("properties", {})
                return {
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "display_name": props.get("label", q),
                    "provider": "BAN",
                }
    except Exception:
        pass

    # --- 2) Nominatim fallback
    try:
        nom_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "fr",
            "q": q,
        }
        r = requests.get(
            nom_url,
            params=params,
            timeout=12,
            headers={
                "User-Agent": f"LYRAE/1.0 ({CONTACT_EMAIL})",
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.6",
            },
        )

        if r.status_code != 200:
            # IMPORTANT : on ne cache pas l'échec, et on laisse l'appelant gérer l'affichage
            return {"__error__": True, "status": r.status_code, "text": r.text[:300], "provider": "Nominatim"}

        data = r.json()
        if not data:
            return None
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        disp = data[0].get("display_name", q)
        return {"lat": lat, "lon": lon, "display_name": disp, "provider": "Nominatim"}

    except Exception:
        return None



def render_map(lat: float, lon: float, zoom: int = 14):
    map_id = f"map_{abs(hash((round(lat,6), round(lon,6), int(zoom))))}"
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin=""
        />
        <script
          src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""
        ></script>
        <style>
          html, body {{ margin:0; padding:0; background:transparent; }}
          #{map_id} {{
            width: 100%;
            height: 420px;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 22px rgba(0,0,0,.12);
            border: 1px solid rgba(14,59,53,.12);
          }}
        </style>
      </head>
      <body>
        <div id="{map_id}"></div>
        <script>
          const map = L.map("{map_id}", {{
            zoomControl: true,
            attributionControl: true,
          }}).setView([{lat}, {lon}], {int(zoom)});

          L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            maxZoom: 19,
            detectRetina: true,
            updateWhenIdle: true,
            keepBuffer: 4,
            attribution: '&copy; OpenStreetMap contributors'
          }}).addTo(map);

          L.marker([{lat}, {lon}]).addTo(map);
        </script>
      </body>
    </html>
    """
    components.html(html, height=440)



# ============================================================
# INIT SESSION (IMPORTANT: setdefault avant widgets)
# ============================================================
st.session_state.setdefault("page", "home")
st.session_state.setdefault("geo", None)
st.session_state.setdefault("risk_class", None)
st.session_state.setdefault("horse_name", "CHEVAL_1")
st.session_state.setdefault("addr_num", "")
st.session_state.setdefault("addr_street", "")
st.session_state.setdefault("addr_city", "")
st.session_state.setdefault("addr_cp", "")
st.session_state.setdefault("last_result", None)


# ============================================================
# TOPBAR
# ============================================================
st.markdown(
    f"""
    <div class="lyrae-topbar">
      <div class="lyrae-topbar-inner">
        <div class="lyrae-brand">
          <div class="lyrae-logo" title="{APP_BRAND}">
            <img src="{MINI_LOGO_URL}" alt="{APP_BRAND}">
          </div>
          <span>{APP_BRAND}</span>
        </div>
        <div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


page = st.session_state.get("page", "home")


# ============================================================
# SIDEBAR: chemins + reset
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    default_model = str(Path(__file__).with_name(MODEL_DEFAULT))
    default_meta  = str(Path(__file__).with_name(META_DEFAULT))
    model_path = st.text_input("Chemin modèle .cbm", value=default_model)
    meta_path  = st.text_input("Chemin meta .json", value=default_meta)

    st.caption("Astuce : place le modèle et le meta dans le même dossier que ce fichier.")

    if st.button("🔄 Reset (session)", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ============================================================
# LOAD MODEL + META
# ============================================================
try:
    model, meta, feature_cols, cat_cols, factor_levels, cat_idx = load_model_and_meta(model_path, meta_path)
except Exception as e:
    st.error(f"Impossible de charger modèle/meta: {e}")
    st.stop()


# ============================================================
# Vérification colonnes vs XLSX
# ============================================================
xlsx_cols = load_xlsx_columns(REF_XLSX_URL, sheet=REF_XLSX_SHEET)

with st.sidebar:
    st.subheader("✅ Contrôle des colonnes")
    if xlsx_cols and isinstance(xlsx_cols[0], str) and xlsx_cols[0].startswith("__ERROR__:"):
        st.warning("Impossible de lire le XLSX de référence.")
        st.caption(xlsx_cols[0])
    else:
        ref_set = {c for c in xlsx_cols if c and c not in REF_XLSX_IGNORE}
        feat_set = set(feature_cols)

        missing_in_model = sorted(ref_set - feat_set)
        extra_in_model   = sorted(feat_set - ref_set)

        if not missing_in_model and not extra_in_model:
            st.success("OK : colonnes modèle = colonnes XLSX.")
        else:
            st.error("⚠️ Mismatch colonnes modèle vs XLSX.")
            with st.expander("Détails"):
                if missing_in_model:
                    st.write("**Dans XLSX mais pas dans feature_cols :**")
                    st.code("\n".join(missing_in_model))
                if extra_in_model:
                    st.write("**Dans feature_cols mais pas dans XLSX :**")
                    st.code("\n".join(extra_in_model))

analysis_cols_set = set(analysis_cols)
results_analysis_set = set([c for c in RESULTS_ANALYSIS_COLS if c in feature_cols])


# ============================================================
# QUESTIONS
# ============================================================
QUESTION = {
    "Age_du_cheval": "Quel est l’âge du cheval (années) ?",
    "Sexe": "Quel est le sexe du cheval ?",
    "Type_de_cheval": "Quel est le type de cheval ?",
    "Season": "Saison au moment de la consultation ?",

    "Exterieur_vegetalisé": "Le cheval a-t-il accès à un extérieur végétalisé ?",
    "Exterieur_vegetalise": "Le cheval a-t-il accès à un extérieur végétalisé ?",
    "Freq_acces_exterieur_sem": "Combien de sorties par semaine (accès extérieur) ?",
    "Tiques_semaines_précédentes": "Des tiques ont-elles été observées ces dernières semaines ?",

    "Examen_clinique": "Un examen clinique a-t-il été réalisé ?",

    "Abattement": "Présente-t-il de l’abattement ?",
    "Mauvaise_performance": "Présente-t-il une baisse de performance ?",
    "Douleurs_diffuses": "Présente-t-il des douleurs diffuses ?",
    "Boiterie": "Présente-t-il une boiterie ?",

    "Meningite": "Suspicion de méningite ?",
    "Radiculonevrite": "Suspicion de radiculonévrite ?",
    "Troubles_de_la_demarche": "Troubles de la démarche ?",
    "Dysphagie": "Dysphagie ?",
    "Fasciculations_musculaires": "Fasciculations musculaires ?",

    "Uveite_bilaterale": "Uvéite bilatérale ?",
    "Cecite_avec_cause_inflammatoire": "Cécité avec cause inflammatoire suspectée ?",
    "Synechies": "Synéchies ?",
    "Atrophie": "Atrophie ?",
    "Dyscories": "Dyscories ?",
    "Myosis": "Myosis ?",

    "Synovite_avec_epanchement_articulaire": "Synovite avec épanchement articulaire ?",
    "Pseudolyphome_cutane": "Pseudolymphome cutané ?",
    "Pododermatite": "Pododermatite ?",

    "piroplasmose_neg": "Piroplasmose exclue (test négatif) ?",
    "ehrlichiose_neg": "Ehrlichiose exclue (test négatif) ?",
    "ehrlichiose_negatif": "Ehrlichiose exclue (test négatif) ?",
    "Bilan_sanguin_normal": "Bilan sanguin normal ?",
    "NFS_normale": "NFS normale ?",

    "Parametres_musculaires_normaux": "Paramètres musculaires normaux ?",
    "Parametres_renaux_normaux": "Paramètres rénaux normaux ?",
    "Parametres_hepatiques_normaux": "Paramètres hépatiques normaux ?",

    "SAA_normal": "SAA normale ?",
    "Fibrinogène_normal": "Fibrinogène normal ?",

    "ELISA_pos": "ELISA positif ?",
    "ELISA_OspA_pos": "ELISA OspA positif ?",
    "ELISA_OspF_pos": "ELISA OspF positif ?",
    "ELISA_p39": "ELISA p39 positif ?",
    "WB_pos": "Western Blot positif ?",
    "PCR_sang_pos": "PCR sang positive ?",
    "SNAP_C6_pos": "SNAP C6 positif ?",
    "IFAT_pos": "IFAT positif ?",

    "PCR_LCR_pos": "PCR LCR positive ?",
    "PCR_synoviale_pos": "PCR synoviale positive ?",
    "PCR_peau_pos": "PCR peau positive ?",
    "PCR_humeur_aqueuse_pos": "PCR humeur aqueuse positive ?",
    "PCR_tissu_nerveux_pos": "PCR tissu nerveux positif ?",
    "PCR_liquide_articulaire_pos": "PCR liquide articulaire positive ?",
    "LCR_pleiocytose": "LCR : pléiocytose ?",
    "LCR_proteines_augmentees": "LCR : protéines augmentées ?",

    "IHC_tissulaire_pos": "Immunohistochimie tissulaire positive ?",
    "Coloration_argent_pos": "Coloration argent positive ?",
    "FISH_tissulaire_pos": "FISH tissulaire positive ?",
    "CVID": "CVID ?",
    "Hypoglobulinemie": "Hypogammaglobulinémie ?",
}

YES_NO_OPTS = ["Oui", "Non"]

def has(col):
    return col in feature_cols

def question_label(col: str) -> str:
    return QUESTION.get(col, col)

def input_widget(col: str, key: str):
    if not has(col):
        return None

    label = question_label(col)

    if col == "Season":
        season_order = ["printemps", "été", "automne", "hiver"]
        lv_meta = [str(x) for x in factor_levels.get(col, [])] if col in cat_cols else []
        if lv_meta:
            def _rank(s):
                s2 = str(s).strip().lower()
                return season_order.index(s2) if s2 in season_order else 999
            options = sorted(lv_meta, key=_rank)
        else:
            options = season_order
        choice = st.selectbox(label, options=options, index=None, placeholder="Sélectionner…", key=key)
        return pd.NA if choice is None else choice

    if col == "Freq_acces_exterieur_sem":
        v = st.number_input(label, min_value=0, max_value=7, step=1, value=None, key=key)
        return pd.NA if v is None else v

    if col in cat_cols:
        lv = factor_levels.get(col, [])
        choice = st.selectbox(label, options=[str(x) for x in lv], index=None, placeholder="Sélectionner…", key=key)
        return pd.NA if choice is None else choice

    bin_like = (
        col.endswith(("_neg", "_normal", "_normale")) or
        col.startswith(("ELISA", "WB", "PCR", "SNAP", "IFAT")) or
        col in (
            "Examen_clinique","Tiques_semaines_précédentes",
            "Meningite","Radiculonevrite","Troubles_de_la_demarche","Dysphagie","Fasciculations_musculaires",
            "Uveite_bilaterale","Cecite_avec_cause_inflammatoire","Synechies","Atrophie","Dyscories","Myosis",
            "Synovite_avec_epanchement_articulaire","Pseudolyphome_cutane","Pododermatite",
            "Abattement","Mauvaise_performance","Douleurs_diffuses","Boiterie",
            "CVID","Hypoglobulinemie","LCR_pleiocytose","LCR_proteines_augmentees",
            "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos",
            "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux"
        )
    )
    if bin_like:
        choice = st.selectbox(label, options=YES_NO_OPTS, index=None, placeholder="Choisir…", key=key)
        return pd.NA if choice is None else choice

    raw = st.text_input(label, value="", placeholder="Laisser vide si inconnu", key=key)
    return pd.NA if raw.strip() == "" else raw.strip()


# ============================================================
# PAGES
# ============================================================
if page == "home":
    st.markdown(
        f"""
        <div class="lyrae-hero">
          <h1>{APP_TITLE}</h1>
          <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="lyrae-illustration">
          <img src="{HERO_IMAGE_URL}" alt="LYRAE" style="width:100%; height:auto; display:block;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="lyrae-cta-wrap">', unsafe_allow_html=True)
    if st.button("Commencer une évaluation clinique  ➜", use_container_width=True):
        st.session_state["page"] = "eval"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="lyrae-disclaimer">
          Cet outil est une aide à la décision, non un dispositif médical autonome.
          Il ne remplace ni l’examen clinique ni le jugement du vétérinaire.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

if page == "methodo":
    st.markdown(f"<div class='lyrae-page-title'>Méthodologie</div>", unsafe_allow_html=True)
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3>Principe général</h3>
        <p>
        LYRAE applique un modèle CatBoost entraîné sur un jeu de données structuré.
        La sortie est une probabilité de Lyme, traduite en catégories d’aide à la décision.
        </p>
        <h3>Risque géographique</h3>
        <p>
        Si <code>Classe_de_risque</code> existe, elle est remplie automatiquement via un raster (3 classes).
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if page == "sources":
    st.markdown(f"<div class='lyrae-page-title'>Sources scientifiques</div>", unsafe_allow_html=True)
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3>À compléter</h3>
        <p>Ajoute ici les références (format APA + DOI si possible).</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if page == "project":
    st.markdown(f"<div class='lyrae-page-title'>Projet RESOLVE</div>", unsafe_allow_html=True)
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3>Contexte</h3>
        <p>Page projet : objectifs, partenaires, modalités, contact.</p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# EVALUATION
# ============================================================
st.markdown(f"<div class='lyrae-page-title'>Évaluation clinique</div>", unsafe_allow_html=True)

top_left, top_right = st.columns([1.2, 0.8])
with top_left:
    if st.button("⬅ Retour accueil"):
        st.session_state["page"] = "home"
        st.rerun()
with top_right:
    if st.button("🪦 Réinitialiser le formulaire", use_container_width=True):
        for k in ["geo","risk_class","horse_name","addr_num","addr_street","addr_city","addr_cp","last_result"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

TAB_LABELS = [
    "Identité",
    "Contexte & exposition",
    "Diagnostic d'exclusion",
    "Signes cliniques",
    "Résultats d'analyse",
]

# Sélecteur contrôlé (segmented si dispo, sinon radio horizontal)
try:
    active_tab = st.segmented_control(
        "",
        options=TAB_LABELS,
        default=TAB_LABELS[0],
        key="active_tab",
    )
except Exception:
    active_tab = st.radio(
        "",
        TAB_LABELS,
        horizontal=True,
        key="active_tab",
        label_visibility="collapsed",
    )

STEP_MAP = {name: i + 1 for i, name in enumerate(TAB_LABELS)}
step = STEP_MAP.get(active_tab, 1)


inputs: dict = {}

# Aliases (accents)
ALIASES = {}
if has("Exterieur_vegetalisé") and not has("Exterieur_vegetalise"):
    ALIASES["Exterieur_vegetalise"] = "Exterieur_vegetalisé"
if has("Exterieur_vegetalise") and not has("Exterieur_vegetalisé"):
    ALIASES["Exterieur_vegetalisé"] = "Exterieur_vegetalise"

def put(col: str, value):
    if col in feature_cols:
        inputs[col] = value
    elif col in ALIASES and ALIASES[col] in feature_cols:
        inputs[ALIASES[col]] = value


if active_tab == "Identité":
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="lyrae-card-header">
          <div>
            <h3 style="margin:0;"> Identité du cheval</h3>
            <div style="margin-top:6px; color:#6d7a79; font-weight:700;">
              Renseigne ce que tu sais — le reste peut rester vide.
            </div>
          </div>
          <div class="lyrae-mini-pill">Étape {step} / 5</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ✅ Nouveau layout : formulaire à gauche, photo à droite
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        horse_name = st.text_input(
    "Nom du cheval",
    value=st.session_state.get("horse_name", "CHEVAL_1"),
    placeholder="Ex: TAGADA",
    key="horse_name"
)


        # Âge puis Type EN COLONNE (stacked)
        if has("Age_du_cheval"):
            put("Age_du_cheval", input_widget("Age_du_cheval", key="id_Age_du_cheval"))

        if has("Type_de_cheval"):
            put("Type_de_cheval", input_widget("Type_de_cheval", key="id_Type_de_cheval"))


        if has("Season"):
            put("Season", input_widget("Season", key="id_Season"))

        if has("Sexe"):
            put("Sexe", input_widget("Sexe", key="id_Sexe"))


    with right:
        # Option A (simple)
        st.image(
            "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/equiphoto.png",
            use_container_width=True
        )

        # Option B (si tu veux la même “carte” que ton hero, décommente ça)
        # st.markdown("<div class='lyrae-illustration'>", unsafe_allow_html=True)
        # st.image("https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/equiphoto.png", use_container_width=True)
        # st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)





elif active_tab == "Contexte & exposition":
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="lyrae-card-header">
          <div>
            <h3 style="margin:0;">🌿 Contexte & exposition</h3>
            <div style="margin-top:6px; color:#6d7a79; font-weight:700;">
              Exposition aux tiques, environnement, localisation.
            </div>
          </div>
          <div class="lyrae-mini-pill">Étape {step} / 5</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        if has("Tiques_semaines_précédentes"):
            put("Tiques_semaines_précédentes", input_widget("Tiques_semaines_précédentes", key="ctx_Tiques_semaines_précédentes"))

    with right:
        if has("Exterieur_vegetalisé") or has("Exterieur_vegetalise"):
            col_ui = "Exterieur_vegetalisé" if has("Exterieur_vegetalisé") else "Exterieur_vegetalise"
            put(col_ui, input_widget(col_ui, key="ctx_Exterieur_vegetalise"))
        if has("Freq_acces_exterieur_sem"):
            put("Freq_acces_exterieur_sem", input_widget("Freq_acces_exterieur_sem", key="ctx_Freq_acces_exterieur_sem"))

    # ============================================================
    # Localisation du cheval (RESTE IDENTIQUE)
    # ============================================================
    st.markdown("---")
    st.markdown("<h3 style='margin-top:6px;'>Localisation du cheval</h3>", unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns([0.22, 0.78, 0.4, 0.4], gap="small")
    with a1:
        num = st.text_input(
            "Numéro",
            value=st.session_state.get("addr_num", ""),
            placeholder="N°",
            key="addr_num",
        )
    with a2:
        street = st.text_input(
            "Rue",
            value=st.session_state.get("addr_street", ""),
            placeholder="Rue / voie",
            key="addr_street",
        )
    with a3:
        city = st.text_input(
            "Ville",
            value=st.session_state.get("addr_city", ""),
            placeholder="Ville",
            key="addr_city",
        )
    with a4:
        cp = st.text_input(
            "Code postal",
            value=st.session_state.get("addr_cp", ""),
            placeholder="CP",
            key="addr_cp",
        )

    locate_col, _ = st.columns([0.34, 0.66])
    with locate_col:
        do_locate = st.button("Localiser sur la carte", use_container_width=True)

    if "geo" not in st.session_state:
        st.session_state["geo"] = None
    if "risk_class" not in st.session_state:
        st.session_state["risk_class"] = None

    if do_locate:
        full_address = " ".join(
            [str(x).strip() for x in [num, street, cp, city] if str(x).strip() != ""]
        ).strip()

        if full_address == "":
            st.session_state["geo"] = None
            st.session_state["risk_class"] = None
            st.warning("Adresse incomplète — renseigne au minimum rue + ville (et idéalement le code postal).")
        else:
            geo_tmp = geocode_address(full_address)

            if isinstance(geo_tmp, dict) and geo_tmp.get("__error__"):
                st.session_state["geo"] = None
                st.session_state["risk_class"] = None
                st.warning(f"Impossible de localiser l’adresse (HTTP {geo_tmp.get('status')}).")
            elif geo_tmp is None:
                st.session_state["geo"] = None
                st.session_state["risk_class"] = None
                st.warning("Adresse non trouvée. Essaye d’ajouter le code postal ou de simplifier l’adresse.")
            else:
                st.session_state["geo"] = geo_tmp

                st.session_state["risk_class"] = risk_class_from_geo(
                    lat_wgs84=geo_tmp["lat"],
                    lon_wgs84=geo_tmp["lon"],
                    factor_levels=factor_levels,
                )

                rc = st.session_state["risk_class"] or "inconnu"
                st.success(f"✅ Localisation effectuée — classe de risque : **{rc}**")

    geo = st.session_state.get("geo", None)
    if geo is not None:
        render_map(geo["lat"], geo["lon"], zoom=14)
    else:
        render_map(46.603354, 1.888334, zoom=5)

    st.markdown("</div>", unsafe_allow_html=True)



elif active_tab == "Diagnostic d'exclusion":
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="lyrae-card-header">
          <div>
            <h3 style="margin:0;">🧪 Diagnostic d'exclusion</h3>
            <div style="margin-top:6px; color:#6d7a79; font-weight:700;">
              Tests négatifs et éléments biologiques orientant vers d'autres causes.
            </div>
          </div>
          <div class="lyrae-mini-pill">Étape {step} / 5</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if has("Examen_clinique"):
            put("Examen_clinique", input_widget("Examen_clinique", key="excl_Examen_clinique"))
    with col2:
        st.caption("")

    col3, col4 = st.columns(2)
    with col3:
        for c in ["piroplasmose_neg", "ehrlichiose_neg", "ehrlichiose_negatif"]:
            if has(c):
                put(c, input_widget(c, key=f"excl_{c}"))
    with col4:
        st.caption("")

    col5, col6 = st.columns(2)
    with col5:
        for c in ["Bilan_sanguin_normal","NFS_normale","SAA_normal","Fibrinogène_normal"]:
            if has(c):
                put(c, input_widget(c, key=f"excl_{c}"))
    with col6:
        for c in ["Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux"]:
            if has(c):
                put(c, input_widget(c, key=f"excl_{c}"))

    st.markdown("</div>", unsafe_allow_html=True)


elif active_tab == "Signes cliniques":
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="lyrae-card-header">
          <div>
            <h3 style="margin:0;">🩺 Signes cliniques</h3>
            <div style="margin-top:6px; color:#6d7a79; font-weight:700;">
              Signes généraux, neurologiques, oculaires, articulaires, cutanés.
            </div>
          </div>
          <div class="lyrae-mini-pill">Étape {step} / 5</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        for c in ["Abattement","Mauvaise_performance"]:
            if has(c):
                put(c, input_widget(c, key=f"sg_{c}"))
    with col2:
        for c in ["Douleurs_diffuses","Boiterie"]:
            if has(c):
                put(c, input_widget(c, key=f"sg_{c}"))

    col3, col4 = st.columns(2)
    with col3:
        for c in ["Meningite","Radiculonevrite","Troubles_de_la_demarche"]:
            if has(c):
                put(c, input_widget(c, key=f"sn_{c}"))
    with col4:
        for c in ["Dysphagie","Fasciculations_musculaires"]:
            if has(c):
                put(c, input_widget(c, key=f"sn_{c}"))

    col5, col6 = st.columns(2)
    with col5:
        for c in ["Uveite_bilaterale","Cecite_avec_cause_inflammatoire","Synechies"]:
            if has(c):
                put(c, input_widget(c, key=f"so_{c}"))
    with col6:
        for c in ["Atrophie","Dyscories","Myosis"]:
            if has(c):
                put(c, input_widget(c, key=f"so_{c}"))

    col7, col8 = st.columns(2)
    with col7:
        if has("Synovite_avec_epanchement_articulaire"):
            put("Synovite_avec_epanchement_articulaire", input_widget("Synovite_avec_epanchement_articulaire", key="sa_Synovite_avec_epanchement_articulaire"))
    with col8:
        st.caption("")

    col9, col10 = st.columns(2)
    with col9:
        for c in ["Pseudolyphome_cutane","Pododermatite"]:
            if has(c):
                put(c, input_widget(c, key=f"sc_{c}"))
    with col10:
        st.caption("")

    already = set(inputs.keys())
    extra_candidates = [
        c for c in feature_cols
        if c not in already
        and not c.endswith("_missing_code")
        and c not in results_analysis_set
        and c != "Classe_de_risque"
    ]
    if extra_candidates:
        st.markdown("---")
        st.markdown("<h3>Autres variables disponibles (modèle)</h3>", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        for i, c in enumerate(extra_candidates):
            target = colA if i % 2 == 0 else colB
            with target:
                put(c, input_widget(c, key=f"extra_{c}"))

    st.markdown("</div>", unsafe_allow_html=True)

elif active_tab == "Résultats d'analyse":
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="lyrae-card-header">
          <div>
            <h3 style="margin:0;">📊 Résultats d'analyse</h3>
            <div style="margin-top:6px; color:#6d7a79; font-weight:700;">
              Sérologies, PCR et autres résultats utiles au modèle.
            </div>
          </div>
          <div class="lyrae-mini-pill">Étape {step} / 5</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cols_left, cols_right = st.columns(2)
    for i, c in enumerate([c for c in RESULTS_ANALYSIS_COLS if has(c)]):
        target = cols_left if i % 2 == 0 else cols_right
        with target:
            put(c, input_widget(c, key=f"res_{c}"))

    st.markdown("---")

    # (ton bouton submit + logique restent EXACTEMENT comme avant, juste sous cet onglet)
    submitted = st.button("Lancer l'aide au diagnostic 🐎", use_container_width=True)

    if submitted:
        with st.spinner("🐎 Le cheval galope… Analyse en cours…"):
            time.sleep(0.25)

            if has("Classe_de_risque"):
                auto_risk = st.session_state.get("risk_class", None)
                inputs["Classe_de_risque"] = pd.NA if (auto_risk is None or str(auto_risk).strip() == "") else auto_risk

            X = build_template(feature_cols)
            X = apply_inputs_to_template(X, inputs)

            X = fill_missing_code_like_R(X, set(analysis_cols))
            X = coerce_like_train_python(X, feature_cols, cat_cols, factor_levels)

            # ✅ Sécurisation CatBoost : pas de pd.NA dans les cat features
            X_cb = X.copy()
            for c in cat_cols:
                if c in X_cb.columns:
                    X_cb[c] = X_cb[c].astype("string").fillna("__MISSING__").astype(str)

            pool_one = Pool(X_cb, cat_features=cat_idx)
            p_one = float(model.predict_proba(pool_one)[:, 1][0])

            cat = cat_from_p_like_R(p_one)

        marker_left = int(max(0, min(100, round(p_one * 100))))

        st.markdown(
            f"""
            <div class="lyrae-result" style="background:{cat_color(cat)};">
              {cat}
              <div class="lyrae-scale">
                <div class="lyrae-marker" style="left:{marker_left}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        missing_feats = []
        for c in feature_cols:
            if c.endswith("_missing_code"):
                continue
            if pd.isna(X.at[0, c]):
                missing_feats.append(c)

        with st.expander("🔎 Détails (valeurs manquantes / aperçu des features)"):
            st.write(f"Variables manquantes (sur {len(feature_cols)} features): **{len(missing_feats)}**")
            if missing_feats:
                st.code("\n".join(missing_feats[:200]))
                if len(missing_feats) > 200:
                    st.caption(f"... +{len(missing_feats)-200} autres")
            st.dataframe(X, use_container_width=True)

    last = st.session_state.get("last_result", None)
    if last is not None:
        st.markdown("---")
        st.markdown("<h3>Exporter le cas</h3>", unsafe_allow_html=True)

        json_bytes = json.dumps(last, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger JSON (cas + résultat)",
            data=json_bytes,
            file_name=f"lyrae_{normalize_key(last.get('horse_name','CHEVAL'))}_case.json",
            mime="application/json",
            use_container_width=True
        )

        flat = {}
        flat["horse_name"] = last.get("horse_name")
        flat["probability"] = last.get("probability")
        flat["category"] = last.get("category")
        flat["risk_class"] = last.get("risk_class")
        geo = last.get("geo") or {}
        flat["geo_lat"] = geo.get("lat")
        flat["geo_lon"] = geo.get("lon")
        flat["geo_display_name"] = geo.get("display_name")
        for k, v in (last.get("inputs") or {}).items():
            flat[k] = v

        df_out = pd.DataFrame([flat])
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Télécharger CSV (1 ligne)",
            data=csv_bytes,
            file_name=f"lyrae_{normalize_key(last.get('horse_name','CHEVAL'))}_case.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)




















