import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool

# ============================================================
# LYRAE / RESOLVE — Streamlit predictor (CatBoost)
# - lit un modèle CatBoost .cbm
# - lit un meta .json (feature_cols, cat_cols, factor_levels)
# - UI retravaillée (landing page + parcours)
# - champ vide => NA
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
# Page config + CSS (style proche de ta maquette)
# ============================================================
st.set_page_config(page_title=f"{APP_BRAND} — {APP_TITLE}", layout="wide")

CSS = """
<style>
/* ----- hide streamlit chrome ----- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}  /* barre streamlit haut */

/* ----- global ----- */
:root{
  --lyrae-green-900:#0e3b35;
  --lyrae-green-800:#154b43;
  --lyrae-green-700:#1f5a51;
  --lyrae-cream:#f4f2ed;
  --lyrae-ink:#1d2a2a;
  --lyrae-muted:#5c6b6a;
  --lyrae-accent:#b06a2a;
  --lyrae-accent-2:#d08a45;
  --card:#ffffffcc;
  --shadow: 0 10px 30px rgba(0,0,0,.12);
  --shadow-soft: 0 8px 22px rgba(0,0,0,.10);
  --radius: 18px;
}

.stApp{
  background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--lyrae-cream) 60%, #efece6 100%);
  color: var(--lyrae-ink);
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
  background: linear-gradient(180deg, var(--lyrae-green-900) 0%, var(--lyrae-green-800) 100%);
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
  font-weight: 700;
  letter-spacing: .5px;
  font-size: 20px;
}
.lyrae-mark{
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #f3d6b5 0%, #d9a971 35%, #a86a2c 100%);
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow: 0 10px 20px rgba(0,0,0,.2) inset;
}
.lyrae-mark svg{opacity:.95}

.lyrae-right{
  display:flex;
  align-items:center;
  gap: 14px;
  color: rgba(255,255,255,.9);
  font-size: 14px;
}
.lyrae-pill{
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.14);
}

/* make selectbox look like a pill */
div[data-testid="stSelectbox"] > div{
  background: rgba(255,255,255,.10) !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,.14) !important;
}
div[data-testid="stSelectbox"] *{
  color: rgba(255,255,255,.95) !important;
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
  font-weight: 750;
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
.lyrae-cta{
  width: min(520px, 95%);
  padding: 16px 18px;
  border-radius: 14px;
  background: linear-gradient(180deg, var(--lyrae-accent-2) 0%, var(--lyrae-accent) 100%);
  color: white;
  font-weight: 800;
  font-size: 18px;
  box-shadow: 0 12px 24px rgba(176,106,42,.30);
  border: 2px solid rgba(255,255,255,.35);
  text-align:center;
}
.lyrae-cta small{
  display:block;
  margin-top: 3px;
  opacity: .85;
  font-weight: 600;
  font-size: 12px;
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

/* ----- evaluation layout ----- */
.lyrae-page-title{
  margin: 26px 0 6px 0;
  font-size: 28px;
  font-weight: 760;
  color: #20424b;
}
.lyrae-page-sub{
  margin: 0 0 18px 0;
  color: #5b6b6a;
}

.lyrae-card{
  background: var(--card);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 18px 18px 8px 18px;
}
.lyrae-card h3{
  margin: 0 0 10px 0;
  font-size: 18px;
  font-weight: 750;
  color:#24474f;
}
.lyrae-chip{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(14,59,53,.08);
  border: 1px solid rgba(14,59,53,.12);
  color: rgba(14,59,53,.95);
  font-weight: 650;
  font-size: 13px;
  margin-bottom: 10px;
}

/* buttons (streamlit) */
.stButton > button{
  border-radius: 12px !important;
  padding: 0.75rem 1.1rem !important;
  font-weight: 750 !important;
}
.stDownloadButton > button{
  border-radius: 12px !important;
  font-weight: 750 !important;
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

    # indices des catégorielles (0-based, côté Pool Python on passe des indices)
    cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    return model, meta, feature_cols, cat_cols, factor_levels, cat_idx

def yn_to_num_if_needed(val, col_is_numeric: bool):
    """
    EXACT esprit de .yn_to_num_if_needed R:
    convertir oui/non -> 1/0 UNIQUEMENT si la colonne est numérique dans le train.
    Ici on ne connaît pas le type train depuis R, donc:
    - on applique cette conversion seulement aux colonnes NON catégorielles (donc supposées numériques)
    - si c'est un texte autre que oui/non -> on laisse, puis to_numeric(coerce) => NA
    """
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
    X = pd.DataFrame([{c: pd.NA for c in feature_cols}])
    return X

def apply_inputs_to_template(X, inputs: dict):
    for k, v in inputs.items():
        if k in X.columns:
            X.at[0, k] = v
    return X

def fill_missing_code_like_R(X: pd.DataFrame, analysis_cols_set: set):
    """
    EXACT logique du bloc 7.B R (cheval unique):
    - init *_missing_code à 0
    - pour chaque *_missing_code: si base NA => 2 si base in analysis_cols else 1
      sinon reste 0
    """
    miss_cols = [c for c in X.columns if c.endswith("_missing_code")]
    if len(miss_cols) == 0:
        return X

    # init 0
    for mc in miss_cols:
        X.at[0, mc] = 0

    for mc in miss_cols:
        base = mc.replace("_missing_code", "")
        if base not in X.columns:
            continue
        is_miss = pd.isna(X.at[0, base])
        if is_miss:
            X.at[0, mc] = 2 if base in analysis_cols_set else 1
    return X

def coerce_like_train_python(X: pd.DataFrame, feature_cols: list, cat_cols: list, factor_levels: dict):
    """
    EXACT esprit de .coerce_like_train R:
    - cat -> categorical avec niveaux train
    - num -> float (numeric double)
    """
    # Catégorielles
    for c in cat_cols:
        if c in X.columns:
            lv = factor_levels.get(c, None)
            X[c] = X[c].astype("string")
            if lv is not None:
                X[c] = pd.Categorical(X[c], categories=lv)
            else:
                X[c] = pd.Categorical(X[c])

    # Numériques = toutes les autres
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for c in num_cols:
        if c not in X.columns:
            continue
        # conversion oui/non -> 1/0 uniquement si "numérique" (donc non cat)
        X[c] = X[c].apply(lambda v: yn_to_num_if_needed(v, col_is_numeric=True))
        X[c] = pd.to_numeric(X[c], errors="coerce")  # non convertible -> NA (comme NA)
    return X

def cat_from_p_like_R(p: float) -> str:
    # EXACT comme ton code R (cheval unique)
    if p < 0.25:
        return "Pas de Lyme ou informations insuffisantes"
    if p < 0.50:
        return "Lyme possible"
    if p < 0.75:
        return "Lyme probable"
    return "Lyme sûr"

# ============================================================
# Topbar (comme maquette)
# ============================================================
st.markdown(
    f"""
    <div class="lyrae-topbar">
      <div class="lyrae-topbar-inner">
        <div class="lyrae-brand">
          <div class="lyrae-mark" title="{APP_BRAND}">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M6 14c2.5-5.5 7-8 12-8 0 0-2 3-2 4 0 0 3 0 4 2 0 0-3 1-4 3 0 0-1 5-6 5-2 0-4-1-4-6z" fill="rgba(14,59,53,.92)"/>
              <circle cx="17.2" cy="7.8" r="0.9" fill="rgba(255,255,255,.9)"/>
            </svg>
          </div>
          <span>{APP_BRAND}</span>
        </div>
        <div class="lyrae-right">
          <span class="lyrae-pill">Outil en veille</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Sidebar: chemins (on garde, mais discret)
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
# Navigation (landing -> evaluation)
# ============================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def goto(page: str):
    st.session_state["page"] = page

# ============================================================
# Widgets (champ vide => NA) — mais en UI plus ordonnée
# ============================================================
def has(col):
    return col in feature_cols

def input_widget(col: str, key: str):
    """
    IMPORTANT:
    - si utilisateur ne renseigne pas -> NA
    - catégorielles: selectbox avec (NA)
    - binaires fréquents: radio (NA/0/1)
    - numériques: text_input (vide => NA) pour éviter toute valeur imposée
    """
    if not has(col):
        return None

    if col in cat_cols:
        lv = factor_levels.get(col, [])
        options = ["(NA)"] + [str(x) for x in lv]
        choice = st.selectbox(col, options=options, index=0, key=key)
        return pd.NA if choice == "(NA)" else choice

    # heuristique binaire
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
        choice = st.radio(col, options=["(NA)", "0", "1"], horizontal=True, index=0, key=key)
        if choice == "(NA)":
            return pd.NA
        return int(choice)

    # numériques / autres : champ texte vide => NA
    raw = st.text_input(col, value="", placeholder="Laisser vide si inconnu", key=key)
    return pd.NA if raw.strip() == "" else raw.strip()

# ============================================================
# HOME (comme ta maquette)
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

    # Illustration (SVG simple, style “bandeau”)
    st.markdown(
        """
        <div class="lyrae-illustration">
          <svg viewBox="0 0 1200 280" width="100%" height="280" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="fog" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="#ffffff" stop-opacity="0.95"/>
                <stop offset="55%" stop-color="#f6f3ec" stop-opacity="0.96"/>
                <stop offset="100%" stop-color="#efece6" stop-opacity="0.98"/>
              </linearGradient>
              <linearGradient id="forest" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0%" stop-color="#cfe1d7"/>
                <stop offset="55%" stop-color="#dbe8d8"/>
                <stop offset="100%" stop-color="#c6d6cf"/>
              </linearGradient>
            </defs>

            <rect x="0" y="0" width="1200" height="280" fill="url(#fog)"/>
            <rect x="0" y="130" width="1200" height="150" fill="url(#forest)" opacity="0.65"/>

            <!-- simplified trees -->
            <g opacity="0.35" fill="#0e3b35">
              <polygon points="80,250 105,190 130,250"/>
              <polygon points="155,250 180,175 205,250"/>
              <polygon points="240,250 265,185 290,250"/>
              <polygon points="980,250 1005,190 1030,250"/>
              <polygon points="1055,250 1080,175 1105,250"/>
              <polygon points="1140,250 1165,185 1190,250"/>
            </g>

            <!-- horse silhouette -->
            <g transform="translate(210,170) scale(1.1)" fill="#0e3b35" opacity="0.90">
              <path d="M-90,55 C-70,10 -20,-10 40,5 C55,8 75,20 85,35 C90,45 88,55 80,60
                       C70,65 60,58 55,52 C52,66 45,78 30,86 C10,98 -20,95 -35,80
                       L-55,95 L-62,90 L-55,75 C-70,74 -85,70 -90,55 Z"/>
              <rect x="-60" y="90" width="10" height="55" rx="2"/>
              <rect x="-20" y="92" width="10" height="55" rx="2"/>
              <rect x="20" y="90" width="10" height="55" rx="2"/>
              <rect x="55" y="85" width="10" height="60" rx="2"/>
            </g>

            <!-- DNA + plus + magnifier tick -->
            <g transform="translate(560,70)">
              <!-- dna -->
              <g transform="translate(0,50)" stroke="#2f6b61" stroke-width="8" fill="none" opacity="0.85">
                <path d="M0,0 C40,30 40,70 0,100"/>
                <path d="M70,0 C30,30 30,70 70,100"/>
                <line x1="15" y1="20" x2="55" y2="20" />
                <line x1="15" y1="50" x2="55" y2="50" />
                <line x1="15" y1="80" x2="55" y2="80" />
              </g>
              <!-- check circle -->
              <circle cx="120" cy="105" r="26" fill="#2f6b61" opacity="0.22"/>
              <path d="M110,105 l8,8 l16,-18" stroke="#2f6b61" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
              <!-- plus -->
              <g transform="translate(170,82)" fill="#b06a2a" opacity="0.95">
                <rect x="0" y="25" width="70" height="18" rx="6"/>
                <rect x="26" y="0" width="18" height="70" rx="6"/>
              </g>
              <!-- magnifier -->
              <g transform="translate(310,38)" opacity="0.92">
                <circle cx="90" cy="100" r="70" fill="none" stroke="#0e3b35" stroke-width="16"/>
                <rect x="140" y="150" width="100" height="26" rx="12" transform="rotate(35 140 150)" fill="#0e3b35"/>
                <!-- tick simplified -->
                <g transform="translate(60,78)" fill="#0e3b35">
                  <ellipse cx="30" cy="30" rx="18" ry="24"/>
                  <circle cx="30" cy="8" r="10"/>
                  <rect x="10" y="28" width="40" height="10" rx="5"/>
                </g>
              </g>
            </g>
          </svg>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CTA (vrai bouton streamlit, mais look “maquette”)
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
# EVALUATION (architecture retravaillée + catégories demandées)
# ============================================================
st.markdown(f"<div class='lyrae-page-title'>Évaluation clinique</div>", unsafe_allow_html=True)
st.markdown("<div class='lyrae-page-sub'>Renseignez ce que vous avez. Laissez vide si inconnu : la variable sera considérée comme NA.</div>", unsafe_allow_html=True)

# ---- barre d’actions haut
top_left, top_mid, top_right = st.columns([1.2, 1.0, 0.8])
with top_left:
    if st.button("⬅ Retour accueil"):
        goto("home")
with top_mid:
    st.markdown("<div class='lyrae-chip'>Modèle CatBoost chargé ✅</div>", unsafe_allow_html=True)
with top_right:
    st.caption("")

# ============================================================
# Catégories / sous-catégories (demandé)
# - Nom du cheval dans Identité (hors modèle)
# - Tout le contexte (y compris exposition) dans Contexte
# - "Clinique / examen" devient Diagnostic d'exclusion
# - Tous les signes cliniques regroupés en sous-catégories
# ============================================================

# ---- Identification (hors modèle)
horse_name_default = st.session_state.get("horse_name", "CHEVAL_1")

# ============================================================
# TABS (propre, ordonné, “outil pro”)
# ============================================================
tab_identity, tab_context, tab_exclusion, tab_signs, tab_biology, tab_advanced = st.tabs([
    "Identité", "Contexte", "Diagnostic d'exclusion", "Signes cliniques", "Biologie spécifique", "Avancé"
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
        horse_name = st.text_input("Nom du cheval", value=horse_name_default, placeholder="Ex: TAGADA")
    with c2:
        st.text_input("Identifiant dossier (optionnel)", value="", placeholder="Ex: RESOLVE-000123")

    st.session_state["horse_name"] = horse_name

    # Variables d'identité si présentes dans feature_cols (sinon non affichées)
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
        # NOTE: "Sexe" est souvent retiré des features (donc n’apparaîtra pas)
        if has("Sexe"):
            inputs["Sexe"] = input_widget("Sexe", key="id_Sexe")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Contexte (inclut exposition)
# -----------------------------
with tab_context:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Contexte & exposition</h3>", unsafe_allow_html=True)

    st.caption("Inclut l’environnement, l’accès extérieur et les indicateurs d’exposition. Laisser vide si inconnu.")

    # Sous-catégorie: Environnement / risque
    st.markdown("#### Environnement / risque")
    colA, colB = st.columns(2)
    with colA:
        if has("Classe de risque"):
            inputs["Classe de risque"] = input_widget("Classe de risque", key="ctx_Classe de risque")
        if has("Classe_de_risque"):
            inputs["Classe_de_risque"] = input_widget("Classe_de_risque", key="ctx_Classe_de_risque")
    with colB:
        if has("Exterieur_vegetalisé"):
            inputs["Exterieur_vegetalisé"] = input_widget("Exterieur_vegetalisé", key="ctx_Exterieur_vegetalisé")
        if has("Freq_acces_exterieur_sem"):
            inputs["Freq_acces_exterieur_sem"] = input_widget("Freq_acces_exterieur_sem", key="ctx_Freq_acces_exterieur_sem")

    # Sous-catégorie: Exposition directe
    st.markdown("#### Exposition directe")
    colC, colD = st.columns(2)
    with colC:
        if has("Tiques_semaines_précédentes"):
            inputs["Tiques_semaines_précédentes"] = input_widget("Tiques_semaines_précédentes", key="ctx_Tiques_semaines_précédentes")
    with colD:
        st.caption("")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Diagnostic d'exclusion (ex-clinique/examen + bilans/exclusions)
# -----------------------------
with tab_exclusion:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Diagnostic d'exclusion</h3>", unsafe_allow_html=True)
    st.caption("Renseignez les éléments disponibles pour exclure d’autres causes (ou objectiver l’état général).")

    # Sous-catégorie: Examen clinique
    st.markdown("#### Examen clinique")
    col1, col2 = st.columns(2)
    with col1:
        if has("Examen_clinique"):
            inputs["Examen_clinique"] = input_widget("Examen_clinique", key="excl_Examen_clinique")
    with col2:
        st.caption("")

    # Sous-catégorie: Co-infections / diagnostics différentiels
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

    # Sous-catégorie: Bilan sanguin / inflammation / organes
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
# Signes cliniques (tous regroupés en sous-catégories)
# -----------------------------
with tab_signs:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Signes cliniques</h3>", unsafe_allow_html=True)
    st.caption("Renseignez uniquement les signes observés. Laisser vide si non évalué / inconnu.")

    # Sous-catégorie: généraux
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

    # Sous-catégorie: neurologiques
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

    # Sous-catégorie: oculaire
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

    # Sous-catégorie: articulaire
    st.markdown("#### Articulaire")
    col7, col8 = st.columns(2)
    with col7:
        if has("Synovite_avec_epanchement_articulaire"):
            inputs["Synovite_avec_epanchement_articulaire"] = input_widget("Synovite_avec_epanchement_articulaire", key="sa_Synovite_avec_epanchement_articulaire")
    with col8:
        st.caption("")

    # Sous-catégorie: cutané / autres
    st.markdown("#### Cutané / autres")
    col9, col10 = st.columns(2)
    with col9:
        for c in ["Pseudolyphome_cutane","Pododermatite"]:
            if has(c):
                inputs[c] = input_widget(c, key=f"sc_{c}")
    with col10:
        st.caption("")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Biologie spécifique (sérologies / PCR / LCR / histo / immuno)
# -----------------------------
with tab_biology:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Biologie spécifique (Bbsl)</h3>", unsafe_allow_html=True)
    st.caption("Examens spécifiques (sérologies, PCR, LCR, histologie, immuno).")

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

# -----------------------------
# Avancé (variables restantes non *_missing_code)
# -----------------------------
with tab_advanced:
    st.markdown("<div class='lyrae-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Avancé</h3>", unsafe_allow_html=True)
    st.caption("Variables supplémentaires provenant directement de feature_cols (optionnel).")

    extra_candidates = [
        c for c in feature_cols
        if c not in inputs
        and not c.endswith("_missing_code")
    ]

    if len(extra_candidates) == 0:
        st.info("Aucune variable supplémentaire à afficher.")
    else:
        # affichage en colonnes
        colA, colB = st.columns(2)
        for i, c in enumerate(extra_candidates):
            target = colA if i % 2 == 0 else colB
            with target:
                inputs[c] = input_widget(c, key=f"adv_{c}")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ACTIONS (en bas)
# ============================================================
st.markdown("---")
action_left, action_right = st.columns([1, 1])

with action_left:
    st.subheader("🔮 Lancer la prédiction")
    submitted = st.button("Prédire P(Lyme)")

with action_right:
    st.subheader("🧾 Rappel")
    st.caption("Champ vide ⇒ NA. *_missing_code auto (analyses=2 MNAR, autres=1 MCAR).")
    st.caption("Catégorie: <0.25 / <0.50 / <0.75 / ≥0.75 (identique à ton script).")

# ============================================================
# Predict (EXACT logique du bloc R "cheval unique")
# ============================================================
if submitted:
    # 1) Template EXACT attendu par CatBoost = toutes les feature_cols
    X = build_template(feature_cols)

    # 2) Remplir ce qu'on a (intersection). Si vide => NA (déjà)
    X = apply_inputs_to_template(X, inputs)

    # 3) Remplir automatiquement les *_missing_code selon NA (NA analyse => 2, sinon 1)
    X = fill_missing_code_like_R(X, analysis_cols_set)

    # 4) Aligner types EXACTEMENT (cat->categorical niveaux, num->float)
    X = coerce_like_train_python(X, feature_cols, cat_cols, factor_levels)

    # 5) Pool + Prédiction CatBoost (Probability)
    pool_one = Pool(X, cat_features=cat_idx)
    p_one = float(model.predict_proba(pool_one)[:, 1][0])

    # 6) Catégorie EXACT comme ton code R (cheval unique)
    cat = cat_from_p_like_R(p_one)

    st.markdown("## 📌 Résultat")
    a, b, c = st.columns(3)
    a.metric("Nom du cheval", st.session_state.get("horse_name", "CHEVAL_1"))
    b.metric("P_Lyme", f"{p_one:.3f}")
    c.metric("Catégorie", cat)

    # aperçu des colonnes non NA
    st.write("### Aperçu des variables réellement prises en compte (non-NA)")
    non_na_cols = X.columns[X.notna().iloc[0]].tolist()
    if len(non_na_cols) == 0:
        st.info("Toutes les variables sont NA (aucune information renseignée).")
    else:
        st.dataframe(X[non_na_cols], use_container_width=True)

    # Export CSV (ligne complète)
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
