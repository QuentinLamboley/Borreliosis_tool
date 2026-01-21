import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
import time

# ============================================================
# LYRAE / RESOLVE — Streamlit predictor (CatBoost)
# - lit un modèle CatBoost .cbm
# - lit un meta .json (feature_cols, cat_cols, factor_levels)
# - UI retravaillée (landing page + parcours)
# - champ non renseigné => NA (sans afficher "(NA)" à l'utilisateur)
# - remplit automatiquement *_missing_code :
#     NA sur analyse => 2 (MNAR)
#     NA hors analyse => 1 (MCAR)
#     non NA => 0
# - prédiction P(Lyme) + classe EXACTEMENT comme ton code :
#     <0.25 / <0.50 / <0.75 / >=0.75
# ============================================================

APP_BRAND = "LYRAE"
APP_TITLE = "Aide au diagnostic de la borréliose de Lyme équine"
APP_SUBTITLE = "Analyse structurée basée sur les données cliniques, biologiques et contextuelles."
MODEL_DEFAULT = "equine_lyme_catboost.cbm"
META_DEFAULT  = "equine_lyme_catboost_meta.json"

# --- Images du repository GitHub (RAW)
HERO_IMAGE_URL  = "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/Lyrae.png"
MINI_LOGO_URL   = "https://raw.githubusercontent.com/QuentinLamboley/Borreliosis_tool/main/minilyrae.png"

# --- Liste des colonnes "analyses" (MNAR si NA) -> EXACT comme ton code
analysis_cols = [
  # bilans / exclusions
  "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
  "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
  "SAA_normal","Fibrinogène_normal",

  # sérologies / PCR sang
  "ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","WB_pos","PCR_sang_pos","SNAP_C6_pos","IFAT_pos",

  # examens ciblés / PCR locales / LCR
  "PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos",
  "PCR_liquide_articulaire_pos","LCR_pleiocytose","LCR_proteines_augmentees",

  # histo / marquages
  "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos",

  # immuno
  "CVID","Hypoglobulinemie"
]

# ============================================================
# Page config + CSS (vert & beige + onglets + inputs plus visibles)
# ============================================================
st.set_page_config(page_title=f"{APP_BRAND} — {APP_TITLE}", layout="wide")

CSS = """
<style>
/* ----- hide streamlit chrome ----- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ----- palette ----- */
:root{
  --g900:#0e3b35;    /* vert sapin */
  --g850:#124640;
  --g800:#154b43;
  --g700:#1f5a51;
  --beige:#f4f2ed;
  --beige2:#efe9df;
  --ink:#1d2a2a;
  --muted:#5c6b6a;
  --accent:#b08b5a;        /* beige doré */
  --accent2:#d2b48c;
  --card:#ffffffcc;
  --shadow: 0 10px 30px rgba(0,0,0,.12);
  --shadow-soft: 0 8px 22px rgba(0,0,0,.10);
  --radius: 18px;
}

/* ----- background ----- */
.stApp{
  background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--beige) 60%, var(--beige2) 100%);
  color: var(--ink);
}
.block-container{
  padding-top: 0rem;
  padding-bottom: 2.2rem;
  max-width: 1100px;
}

/* ----- top bar ----- */
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
  font-weight: 800;
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
.lyrae-logo img{
  width:100%;
  height:100%;
  display:block;
}

/* ----- tabs visibility ----- */
div[data-testid="stTabs"]{
  background: transparent;
}
div[data-testid="stTabs"] button[role="tab"]{
  border-radius: 12px !important;
  padding: 10px 14px !important;
  margin-right: 8px !important;
  background: rgba(14,59,53,.07) !important;
  border: 1px solid rgba(14,59,53,.18) !important;
  color: rgba(14,59,53,.92) !important;
  font-weight: 750 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  background: rgba(14,59,53,.13) !important;
  border: 2px solid rgba(14,59,53,.35) !important;
  box-shadow: 0 8px 18px rgba(0,0,0,.08) !important;
}

/* ----- inputs readability (replace dark/black) ----- */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div,
div[data-testid="stNumberInput"] > div > div {
  border-radius: 12px !important;
  border: 1px solid rgba(14,59,53,.22) !important;
  background: rgba(255,255,255,.78) !important;
}

/* Streamlit selectbox inner control sometimes uses dark bg in some themes:
   force a sapin-tinted background and readable text */
div[data-testid="stSelectbox"] div[role="combobox"]{
  background: rgba(14,59,53,.08) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(14,59,53,.20) !important;
}
div[data-testid="stSelectbox"] *{
  color: rgba(14,59,53,.95) !important;
}

/* Labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
  color: rgba(14,59,53,.92) !important;
  font-weight: 700 !important;
}

/* ----- hero ----- */
.lyrae-hero{
  padding: 56px 0 24px 0;
  text-align: center;
}
.lyrae-hero h1{
  margin: 0;
  font-size: 42px;
  line-height: 1.12;
  font-weight: 780;
  color: #20424b;
}
.lyrae-hero p{
  margin: 14px auto 0 auto;
  max-width: 820px;
  font-size: 17px;
  color: #5b6b6a;
}
.lyrae-illustration{
  margin: 30px auto 24px auto;
  border-radius: 22px;
  background: radial-gradient(900px 260px at 50% 30%, #ffffff 0%, #f7f4ee 45%, #f2efe8 100%);
  box-shadow: var(--shadow-soft);
  overflow:hidden;
  border: 1px solid rgba(0,0,0,.05);
}
.lyrae-cta-wrap{
  display:flex;
  align-items:center;
  justify-content:center;
  margin-top: 18px;
}
.lyrae-disclaimer{
  margin-top: 18px;
  color: #6d7a79;
  font-size: 14px;
}
.lyrae-footerlinks{
  margin-top: 18px;
  color: #6d7a79;
  font-size: 14px;
}
.lyrae-footerlinks span{
  padding: 0 10px;
  opacity:.55;
}

/* ----- cards ----- */
.lyrae-page-title{
  margin: 26px 0 6px 0;
  font-size: 28px;
  font-weight: 780;
  color: #20424b;
}
.lyrae-page-sub{
  margin: 0 0 18px 0;
  color: #5b6b6a;
}
.lyrae-card{
  background: var(--card);
  border: 1px solid rgba(14,59,53,.12);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 18px 18px 10px 18px;
}
.lyrae-card h3{
  margin: 0 0 10px 0;
  font-size: 18px;
  font-weight: 780;
  color:#24474f;
}

/* buttons */
.stButton > button, .stDownloadButton > button{
  border-radius: 12px !important;
  padding: 0.75rem 1.1rem !important;
  font-weight: 780 !important;
  border: 1px solid rgba(14,59,53,.25) !important;
}
.stButton > button{
  background: linear-gradient(180deg, var(--accent2) 0%, var(--accent) 100%) !important;
  color: rgba(14,59,53,.98) !important;
}
.stButton > button:hover{
  filter: brightness(1.02);
}

/* centered action */
.lyrae-action{
  display:flex;
  justify-content:center;
  align-items:center;
  margin-top: 8px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# Helpers (respect logique R)
# ============================================================
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

def yn_to_num_if_needed(val, col_is_numeric: bool):
    if val is None:
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
    if len(miss_cols) == 0:
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

def coerce_like_train_python(X: pd.DataFrame, feature_cols: list, cat_cols: list, factor_levels: dict):
    for c in cat_cols:
        if c in X.columns:
            lv = factor_levels.get(c, None)
            X[c] = X[c].astype("string")
            X[c] = pd.Categorical(X[c], categories=lv) if lv is not None else pd.Categorical(X[c])

    num_cols = [c for c in feature_cols if c not in cat_cols]
    for c in num_cols:
        if c not in X.columns:
            continue

        # Oui/Non -> 1/0 (si applicable)
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

# ============================================================
# Topbar (logo minilyrae.png) — (suppression "Outil en veille")
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

# ============================================================
# Sidebar: chemins
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    default_model = str(Path(__file__).with_name(MODEL_DEFAULT))
    default_meta  = str(Path(__file__).with_name(META_DEFAULT))
    model_path = st.text_input("Chemin modèle .cbm", value=default_model)
    meta_path  = st.text_input("Chemin meta .json", value=default_meta)

# ============================================================
# Load model + meta
# ============================================================
try:
    model, meta, feature_cols, cat_cols, factor_levels, cat_idx = load_model_and_meta(model_path, meta_path)
except Exception as e:
    st.error(f"Impossible de charger modèle/meta: {e}")
    st.stop()

analysis_cols_set = set(analysis_cols)

# ============================================================
# Navigation (landing -> evaluation) — 1 clic + rerun
# ============================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def goto(page: str):
    st.session_state["page"] = page
    st.rerun()

# ============================================================
# Libellés "questions" (Oui/Non) — sans option NA visible
# ============================================================
QUESTION = {
    "Age_du_cheval": "Quel est l’âge du cheval (années) ?",
    "Sexe": "Quel est le sexe du cheval ?",
    "Type_de_cheval": "Quel est le type de cheval ?",
    "Season": "Quelle est la saison au moment de la consultation ?",

    "Classe de risque": "Quel est le niveau de risque (SPF / zone à risque) ?",
    "Classe_de_risque": "Quel est le niveau de risque (autre variable) ?",
    "Exterieur_vegetalisé": "Le cheval a-t-il accès à un extérieur végétalisé ?",
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
    """
    - Pas d'option "(NA)" visible.
    - Si l'utilisateur ne choisit rien => None => NA en arrière-plan.
    - Les variables binaires sont demandées en Oui/Non (pas 0/1).
    """
    if not has(col):
        return None

    label = question_label(col)

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
            "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos"
        )
    )
    if bin_like:
        choice = st.selectbox(label, options=YES_NO_OPTS, index=None, placeholder="Choisir…", key=key)
        return pd.NA if choice is None else choice

    raw = st.text_input(label, value="", placeholder="Laisser vide si inconnu", key=key)
    return pd.NA if raw.strip() == "" else raw.strip()

# ============================================================
# HOME
# ============================================================
if st.session_state["page"] == "home":
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
        goto("eval")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="lyrae-disclaimer">
          Cet outil est une aide à la décision, non un dispositif médical autonome.
        </div>
        <div class="lyrae-footerlinks">
          <a href="#" style="color:#6d7a79; text-decoration:underline;">Méthodologie</a>
          <span>|</span>
          <a href="#" style="color:#6d7a79; text-decoration:underline;">Sources scientifiques</a>
          <span>|</span>
          <a href="#" style="color:#6d7a79; text-decoration:underline;">Projet RESOLVE</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# ============================================================
# EVALUATION
# ============================================================
st.markdown(f"<div class='lyrae-page-title'>Évaluation clinique</div>", unsafe_allow_html=True)

top_left, top_mid, top_right = st.columns([1.2, 1.0, 0.8])
with top_left:
    if st.button("⬅ Retour accueil"):
        goto("home")
with top_mid:
    st.write("")  # (suppression "Modèle prêt ✅")
with top_right:
    st.caption("")

# ============================================================
# Catégories / sous-catégories (suppression onglet "Avancé")
# - le contenu "Avancé" est déplacé en fin de "Signes cliniques"
# ============================================================
tab_identity, tab_context, tab_exclusion, tab_signs, tab_biology = st.tabs([
    "Identité", "Contexte & exposition", "Diagnostic d'exclusion", "Signes cliniques", "Biologie spécifique"
])

inputs = {}

# -----------------------------
# Identité
# -----------------------------
with tab_identity:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Identité du cheval</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        horse_name = st.text_input("Nom du cheval", value=st.session_state.get("horse_name", "CHEVAL_1"), placeholder="Ex: TAGADA")
    with c2:
        st.text_input("Identifiant dossier (optionnel)", value="", placeholder="Ex: RESOLVE-000123")

    st.session_state["horse_name"] = horse_name

    c3, c4 = st.columns(2)
    with c3:
        if has("Age_du_cheval"):
            inputs["Age_du_cheval"] = input_widget("Age_du_cheval", key="id_Age_du_cheval")
    with c4:
        if has("Type_de_cheval"):
            inputs["Type_de_cheval"] = input_widget("Type_de_cheval", key="id_Type_de_cheval")

    c5, c6 = st.columns(2)
    with c5:
        if has("Season"):
            inputs["Season"] = input_widget("Season", key="id_Season")
    with c6:
        if has("Sexe"):
            inputs["Sexe"] = input_widget("Sexe", key="id_Sexe")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Contexte & exposition (mieux organisé comme le screen)
# - 2 colonnes principales
# - à gauche : risque + exposition directe
# - à droite : extérieur + fréquence (stack vertical)
# -----------------------------
with tab_context:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Contexte & exposition</h3>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown("#### Environnement / risque")
        if has("Classe de risque"):
            inputs["Classe de risque"] = input_widget("Classe de risque", key="ctx_Classe de risque")
        if has("Classe_de_risque"):
            inputs["Classe_de_risque"] = input_widget("Classe_de_risque", key="ctx_Classe_de_risque")

        st.markdown("#### Exposition directe")
        if has("Tiques_semaines_précédentes"):
            inputs["Tiques_semaines_précédentes"] = input_widget("Tiques_semaines_précédentes", key="ctx_Tiques_semaines_précédentes")

    with right:
        st.markdown("#### Milieu de vie / accès")
        if has("Exterieur_vegetalisé"):
            inputs["Exterieur_vegetalisé"] = input_widget("Exterieur_vegetalisé", key="ctx_Exterieur_vegetalisé")
        if has("Freq_acces_exterieur_sem"):
            inputs["Freq_acces_exterieur_sem"] = input_widget("Freq_acces_exterieur_sem", key="ctx_Freq_acces_exterieur_sem")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Diagnostic d'exclusion
# -----------------------------
with tab_exclusion:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Diagnostic d'exclusion</h3>", unsafe_allow_html=True)

    st.markdown("#### Examen clinique")
    col1, col2 = st.columns(2)
    with col1:
        if has("Examen_clinique"):
            inputs["Examen_clinique"] = input_widget("Examen_clinique", key="excl_Examen_clinique")
    with col2:
        st.caption("")

    st.markdown("#### Co-infections / diagnostics différentiels")
    col3, col4 = st.columns(2)
    with col3:
        if has("piroplasmose_neg"):
            inputs["piroplasmose_neg"] = input_widget("piroplasmose_neg", key="excl_piroplasmose_neg")
        if has("ehrlichiose_neg"):
            inputs["ehrlichiose_neg"] = input_widget("ehrlichiose_neg", key="excl_ehrlichiose_neg")
        if has("ehrlichiose_negatif"):
            inputs["ehrlichiose_negatif"] = input_widget("ehrlichiose_negatif", key="excl_ehrlichiose_negatif")
    with col4:
        st.caption("")

    st.markdown("#### Bilan sanguin / inflammation / organes")
    col5, col6 = st.columns(2)
    with col5:
        for c in ["Bilan_sanguin_normal","NFS_normale","SAA_normal","Fibrinogène_normal"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"excl_{c}")
    with col6:
        for c in ["Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"excl_{c}")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Signes cliniques + (ex-Avancé)
# -----------------------------
with tab_signs:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Signes cliniques</h3>", unsafe_allow_html=True)

    st.markdown("#### Signes généraux")
    col1, col2 = st.columns(2)
    with col1:
        for c in ["Abattement","Mauvaise_performance"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"sg_{c}")
    with col2:
        for c in ["Douleurs_diffuses","Boiterie"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"sg_{c}")

    st.markdown("#### Neurologique")
    col3, col4 = st.columns(2)
    with col3:
        for c in ["Meningite","Radiculonevrite","Troubles_de_la_demarche"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"sn_{c}")
    with col4:
        for c in ["Dysphagie","Fasciculations_musculaires"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"sn_{c}")

    st.markdown("#### Oculaire")
    col5, col6 = st.columns(2)
    with col5:
        for c in ["Uveite_bilaterale","Cecite_avec_cause_inflammatoire","Synechies"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"so_{c}")
    with col6:
        for c in ["Atrophie","Dyscories","Myosis"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"so_{c}")

    st.markdown("#### Articulaire")
    col7, col8 = st.columns(2)
    with col7:
        if has("Synovite_avec_epanchement_articulaire"):
            inputs["Synovite_avec_epanchement_articulaire"] = input_widget("Synovite_avec_epanchement_articulaire", key="sa_Synovite_avec_epanchement_articulaire")
    with col8:
        st.caption("")

    st.markdown("#### Cutané / autres")
    col9, col10 = st.columns(2)
    with col9:
        for c in ["Pseudolyphome_cutane","Pododermatite"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"sc_{c}")
    with col10:
        st.caption("")

    # --- ex-Avancé : variables supplémentaires intégrées ici, en "questions" comme le reste
    extra_candidates = [
        c for c in feature_cols
        if c not in inputs
        and not c.endswith("_missing_code")
    ]
    if len(extra_candidates) > 0:
        st.markdown("#### Autres informations (si disponibles)")
        colA, colB = st.columns(2)
        for i, c in enumerate(extra_candidates):
            target = colA if i % 2 == 0 else colB
            with target:
                inputs[c] = input_widget(c, key=f"extra_{c}")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Biologie spécifique
# -----------------------------
with tab_biology:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Biologie spécifique (Bbsl)</h3>", unsafe_allow_html=True)

    st.markdown("#### Sérologies")
    col1, col2 = st.columns(2)
    with col1:
        for c in ["ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"bio_{c}")
    with col2:
        for c in ["WB_pos","SNAP_C6_pos","IFAT_pos"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"bio_{c}")

    st.markdown("#### PCR")
    col3, col4 = st.columns(2)
    with col3:
        for c in ["PCR_sang_pos","PCR_LCR_pos","PCR_synoviale_pos","PCR_liquide_articulaire_pos"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"bio_{c}")
    with col4:
        for c in ["PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"bio_{c}")

    st.markdown("#### LCR / Histologie / Immuno")
    col5, col6 = st.columns(2)
    with col5:
        for c in ["LCR_pleiocytose","LCR_proteines_augmentees","IHC_tissulaire_pos"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"bio_{c}")
    with col6:
        for c in ["Coloration_argent_pos","FISH_tissulaire_pos","CVID","Hypoglobulinemie"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"bio_{c}")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ACTIONS (bas) + "cheval qui galope" pendant le chargement
# ============================================================
st.markdown("---")
submitted = st.button("Lancer l'aide au diagnostic 🐎", use_container_width=True)

# ============================================================
# Predict (EXACT logique du bloc R "cheval unique")
# ============================================================
if submitted:
    with st.spinner("🐎 Le cheval galope… Analyse en cours…"):
        # petite pause pour rendre l’animation perceptible même si prédiction rapide
        time.sleep(0.25)

        X = build_template(feature_cols)
        X = apply_inputs_to_template(X, inputs)
        X = fill_missing_code_like_R(X, analysis_cols_set)
        X = coerce_like_train_python(X, feature_cols, cat_cols, factor_levels)

        pool_one = Pool(X, cat_features=cat_idx)
        p_one = float(model.predict_proba(pool_one)[:, 1][0])
        cat = cat_from_p_like_R(p_one)

    st.markdown("## 📌 Résultat")
    a, b, c = st.columns(3)
    a.metric("Nom du cheval", st.session_state.get("horse_name", "CHEVAL_1"))
    b.metric("P_Lyme", f"{p_one:.3f}")
    c.metric("Catégorie", cat)

    st.write("### Aperçu des variables réellement prises en compte (non-NA)")
    non_na_cols = X.columns[X.notna().iloc[0]].tolist()
    if len(non_na_cols) == 0:
        st.info("Aucune information n’a été renseignée.")
    else:
        st.dataframe(X[non_na_cols], use_container_width=True)

    out = X.copy()
    out.insert(0, "Nom_du_Cheval", st.session_state.get("horse_name", "CHEVAL_1"))
    out["P_Lyme"] = p_one
    out["Cat"] = cat
    csv_bytes = out.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Télécharger la ligne (CSV)",
        data=csv_bytes,
        file_name=f"{st.session_state.get('horse_name','CHEVAL_1')}_prediction.csv",
        mime="text/csv"
    )

    st.caption("Cet outil est une aide à la décision, non un dispositif médical autonome.")
