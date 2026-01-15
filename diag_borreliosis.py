import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool

# ============================================================
# RESOLVE — Streamlit predictor (CatBoost)
# - lit un modèle CatBoost .cbm
# - lit un meta .json (feature_cols, cat_cols, factor_levels)
# - formulaire structuré
# - si l'utilisateur ne remplit pas => NA
# - remplit automatiquement *_missing_code :
#     NA sur analyse => 2 (MNAR)
#     NA hors analyse => 1 (MCAR)
#     non NA => 0
# - prédiction P(Lyme) + classe EXACTEMENT comme ton code :
#     <0.25 / <0.50 / <0.75 / >=0.75
# ============================================================

APP_TITLE = "RESOLVE — Prédiction P(Lyme) (CatBoost)"
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

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Formulaire clinique → NA si non renseigné → *_missing_code auto → P(Lyme) + catégorie (<0.25/<0.50/<0.75/>=0.75).")

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
# Sidebar: chemins
# ============================================================
st.sidebar.header("⚙️ Configuration")
default_model = str(Path(__file__).with_name(MODEL_DEFAULT))
default_meta  = str(Path(__file__).with_name(META_DEFAULT))

model_path = st.sidebar.text_input("Chemin modèle .cbm", value=default_model)
meta_path  = st.sidebar.text_input("Chemin meta .json", value=default_meta)

# ============================================================
# Load model + meta
# ============================================================
try:
    model, meta, feature_cols, cat_cols, factor_levels, cat_idx = load_model_and_meta(model_path, meta_path)
except Exception as e:
    st.error(f"Impossible de charger modèle/meta: {e}")
    st.stop()

st.success("✅ Modèle et metadata chargés")

analysis_cols_set = set(analysis_cols)

# ============================================================
# UI: formulaire structuré (champ vide => NA)
# ============================================================
st.subheader("🧾 Formulaire cheval")

def has(col): 
    return col in feature_cols

def input_widget(col: str):
    """
    IMPORTANT:
    - si utilisateur ne renseigne pas -> NA
    - catégorielles: selectbox avec (NA)
    - binaires fréquents: radio (NA/0/1)
    - numériques: number_input avec option vide -> NA
    """
    if not has(col):
        return None

    if col in cat_cols:
        lv = factor_levels.get(col, [])
        options = ["(NA)"] + [str(x) for x in lv]
        choice = st.selectbox(col, options=options, index=0)
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
        choice = st.radio(col, options=["(NA)", "0", "1"], horizontal=True, index=0)
        if choice == "(NA)":
            return pd.NA
        return int(choice)

    # numériques courants
    if col in ("Age_du_cheval", "Freq_acces_exterieur_sem"):
        if col == "Age_du_cheval":
            v = st.number_input(col, min_value=0, max_value=60, value=None, step=1)
        else:
            v = st.number_input(col, min_value=0, max_value=14, value=None, step=1)
        return pd.NA if v is None else float(v)

    # fallback texte libre (si vide => NA)
    v = st.text_input(col, value="")
    return pd.NA if v.strip() == "" else v.strip()

# Groupes d’affichage (ne change rien au modèle, juste l’UI)
GROUPS = [
    ("Identité & contexte", [
        "Age_du_cheval", "Sexe", "Type_de_cheval", "Season"
    ]),
    ("Exposition / environnement", [
        "Classe de risque", "Classe_de_risque",
        "Exterieur_vegetalisé", "Freq_acces_exterieur_sem", "Tiques_semaines_précédentes"
    ]),
    ("Clinique / examen", [
        "Examen_clinique",
        "Abattement", "Mauvaise_performance", "Douleurs_diffuses", "Boiterie"
    ]),
    ("Signes neurologiques / forts", [
        "Meningite", "Radiculonevrite", "Troubles_de_la_demarche", "Dysphagie", "Fasciculations_musculaires"
    ]),
    ("Signes oculaires & cutanés", [
        "Uveite_bilaterale", "Cecite_avec_cause_inflammatoire", "Synechies", "Atrophie", "Dyscories", "Myosis",
        "Pseudolyphome_cutane", "Pododermatite"
    ]),
    ("Articulaire", [
        "Synovite_avec_epanchement_articulaire"
    ]),
    ("Bilans / exclusions", [
        "piroplasmose_neg", "ehrlichiose_neg", "ehrlichiose_negatif",
        "Bilan_sanguin_normal", "NFS_normale",
        "Parametres_musculaires_normaux", "Parametres_renaux_normaux", "Parametres_hepatiques_normaux",
        "SAA_normal", "Fibrinogène_normal"
    ]),
    ("Sérologies / PCR", [
        "ELISA_pos", "ELISA_OspA_pos", "ELISA_OspF_pos", "ELISA_p39",
        "WB_pos", "PCR_sang_pos",
        "SNAP_C6_pos", "IFAT_pos",
        "PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos",
        "PCR_liquide_articulaire_pos",
        "LCR_pleiocytose","LCR_proteines_augmentees",
        "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos",
        "CVID","Hypoglobulinemie"
    ]),
]

with st.form("horse_form", clear_on_submit=False):
    col_left, col_right = st.columns([1, 1])

    with col_left:
        horse_id = st.text_input("ID / Nom du cheval (hors modèle)", value="CHEVAL_1")

    inputs = {}

    for idx, (gname, cols) in enumerate(GROUPS):
        target = col_left if idx % 2 == 0 else col_right
        with target:
            with st.expander(gname, expanded=(gname == "Identité & contexte")):
                for c in cols:
                    if has(c):
                        inputs[c] = input_widget(c)

    # (optionnel UI) variables supplémentaires
    extra_candidates = [
        c for c in feature_cols
        if c not in inputs
        and not c.endswith("_missing_code")
    ]
    show_extra = st.checkbox("Afficher les variables supplémentaires (avancé)", value=False)
    if show_extra:
        st.info("Ces variables viennent directement de feature_cols. Laisse vide = NA.")
        for c in extra_candidates:
            inputs[c] = input_widget(c)

    submitted = st.form_submit_button("🔮 Prédire P(Lyme)")

# ============================================================
# Predict (EXACT logique du bloc R "cheval unique")
# ============================================================
if submitted:
    # 1) Template EXACT attendu par CatBoost = toutes les feature_cols
    X = build_template(feature_cols)

    # 2) Remplir ce qu'on a (intersection). Si vide => NA (déjà)
    X = apply_inputs_to_template(X, inputs)

    # 3) (comportement R: renommages éventuels)
    # Ici on ne renomme rien automatiquement: le formulaire utilise directement les noms du modèle.
    # Si un utilisateur met rien => NA.

    # 4) Remplir automatiquement les *_missing_code selon NA (NA analyse => 2, sinon 1)
    X = fill_missing_code_like_R(X, analysis_cols_set)

    # 5) Aligner types EXACTEMENT (cat->categorical niveaux, num->float)
    X = coerce_like_train_python(X, feature_cols, cat_cols, factor_levels)

    # 6) Pool + Prédiction CatBoost (Probability)
    pool_one = Pool(X, cat_features=cat_idx)
    p_one = float(model.predict_proba(pool_one)[:, 1][0])

    # 7) Catégorie EXACT comme ton code (cheval unique)
    cat = cat_from_p_like_R(p_one)

    st.markdown("---")
    st.subheader("📌 Résultat")

    a, b, c = st.columns(3)
    a.metric("ID", horse_id)
    b.metric("P_Lyme", f"{p_one:.3f}")
    c.metric("Cat", cat)

    # aperçu des colonnes non NA
    st.write("### Aperçu des variables réellement prises en compte (non-NA)")
    non_na_cols = X.columns[X.notna().iloc[0]].tolist()
    st.dataframe(X[non_na_cols])

    # Export CSV (ligne complète)
    out = X.copy()
    out.insert(0, "ID", horse_id)
    out["P_Lyme"] = p_one
    out["Cat"] = cat
    csv_bytes = out.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Télécharger la ligne (CSV)",
        data=csv_bytes,
        file_name=f"{horse_id}_prediction.csv",
        mime="text/csv"
    )

    st.caption("Règles: champ vide => NA. *_missing_code auto (analyses=2 MNAR, autres=1 MCAR). Cat: <0.25/<0.50/<0.75/≥0.75.")
