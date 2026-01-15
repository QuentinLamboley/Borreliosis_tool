# ==============================================================================
# © 2026 Quentin Lamboley. Tous droits réservés.
#
# AVIS DE LICENCE (PROPRIÉTAIRE / RESTRICTIVE)
# Ce code et toute sa documentation associée sont protégés par le
# droit d’auteur. AUCUNE LICENCE n’est accordée, explicitement ou implicitement,
# à quiconque, sauf autorisation écrite préalable de Quentin Lamboley.
#
# INTERDICTIONS (sauf autorisation écrite) :
# - Utiliser, exécuter, déployer, ou intégrer ce Logiciel dans un produit/service
# - Copier, modifier, adapter, traduire, ou créer des œuvres dérivées
# - Distribuer, publier, vendre, sous-licencier, louer, prêter ou partager
# - Rétroconcevoir (reverse engineering) lorsque applicable
#
# AUTORISATIONS :
# Toute utilisation nécessite un accord écrit (licence) émis par Quentin Lamboley.
# Demandes : quentin.lamboley@anses.fr
#
# ABSENCE DE GARANTIE :
# Le Lcode est fourni “TEL QUEL”, sans garantie d’aucune sorte.En aucun cas
# Quentin Lamboley ne pourra être tenu responsable de dommages directs ou
# indirects liés à l’utilisation ou l’impossibilité d’utiliser ce Logiciel.
# ==============================================================================

# ----------------------------------------------------------------------
# Bloc 1 — Dépendances (packages)
# ----------------------------------------------------------------------

# - On vérifie que les packages existent. Sinon on les installe.
# - R6 : pour créer une “classe” (un générateur réutilisable).
# - openxlsx : pour exporter en Excel.
# - jsonlite : pour exporter une config en JSON.

suppressWarnings({
  if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")
  if (!requireNamespace("openxlsx", quietly = TRUE)) install.packages("openxlsx")
  if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
})
library(R6)
library(openxlsx)
library(jsonlite)


# ----------------------------------------------------------------------
# Bloc 2 — Helpers déterministes (aucun hasard)
# ----------------------------------------------------------------------

# Ici on crée des petites fonctions “utilitaires” :
# - cycle_to_n : répéter un motif (ex: 0:7) pour remplir n lignes.
# - set_if_col : modifier une colonne uniquement si elle existe (sécurité).
# - zero_cols : créer/remplir plusieurs colonnes à 0.
# - missingness : poser des NA + écrire un code qui explique POURQUOI c’est manquant.

cycle_to_n <- function(x, n) {
  if (length(x) == 0) stop("cycle_to_n: x vide")
  rep(x, length.out = n)
}

set_if_col <- function(df, col, idx, value) {
  if (col %in% names(df) && length(idx) > 0) df[idx, col] <- value
  df
}

zero_cols <- function(df, cols) {
  for (cc in cols) df[[cc]] <- 0L
  df
}

set_missing_with_code <- function(df, col, idx, code_col, code_value) {
  if (!(col %in% names(df))) return(df)
  if (!(code_col %in% names(df))) return(df)
  if (length(idx) == 0) return(df)
  df[idx, col] <- NA
  df[idx, code_col] <- as.integer(code_value)
  df
}

# MCAR déterministe (pattern fixe) : met NA + code = 1
# - MCAR = “manquant complètement au hasard” (dans l’idée).
# - Ici on simule ça sans hasard car on souhaite que toutes les colonnes non
# non vitales au diagnostic de la maladie dans un contexte données comportent
#des absences.

apply_mcar_pattern <- function(df, col, code_col, every_k) {
  n <- nrow(df)
  idx <- which((seq_len(n) %% every_k) == 0)
  set_missing_with_code(df, col, idx, code_col, 1L)
}

# MNAR déterministe : met NA + code=2 sur un sous-ensemble déterministe
# keep_every_k : garde 1 ligne sur k (donc drop k-1 lignes sur k)
# - MNAR = “manquant pas au hasard” (ex: test non fait car pas d’indication).
# - Ici : sur une liste d’indices candidats, on “garde” 1 sur k, et le reste devient NA + code 2.

apply_mnar_deterministic <- function(df, col, code_col, idx_candidate, keep_every_k = 4) {
  if (length(idx_candidate) == 0) return(df)
  to_drop <- idx_candidate[(seq_along(idx_candidate) %% keep_every_k) != 0]
  set_missing_with_code(df, col, to_drop, code_col, 2L)
}


# ----------------------------------------------------------------------
# Bloc 3 — Classe générateur (R6)
# ----------------------------------------------------------------------

# On construit un “objet générateur”.
# Il sait :
# - generate() : fabriquer le dataset (les règles de cas parfaits)
# - summarize() : faire un mini résumé (nb lignes + taux MCAR/MNAR)
# - save_outputs() : exporter en xlsx/csv/json

EquineLymePerfectDatasetGenerator <- R6Class(
  "EquineLymePerfectDatasetGenerator",
  public = list(
    n_per_class = NULL,
    add_missingness = NULL,
    base_random_state = NULL,
    
    initialize = function(n_per_class = 50, base_random_state = 42, add_missingness = TRUE) {
      self$n_per_class <- n_per_class
      self$add_missingness <- add_missingness
      self$base_random_state <- base_random_state
    },
    
    summarize = function(df) {

      summary <- list(n = nrow(df))
      if ("Lyme_true" %in% names(df)) {
        lt <- df$Lyme_true
        if (is.factor(lt)) lt <- as.character(lt)
        lt_int <- suppressWarnings(as.integer(lt))
        summary$n_lyme_true_1 <- as.integer(sum(lt_int == 1L, na.rm = TRUE))
        summary$n_lyme_true_0 <- as.integer(sum(lt_int == 0L, na.rm = TRUE))
      }
      missing_cols <- names(df)[grepl("_missing_code$", names(df))]
      if (length(missing_cols) > 0) {
        mcar_count <- sum(df[, missing_cols, drop = FALSE] == 1, na.rm = TRUE)
        mnar_count <- sum(df[, missing_cols, drop = FALSE] == 2, na.rm = TRUE)
        total_obs <- nrow(df) * length(missing_cols)
        summary$MCAR_rate <- as.numeric(mcar_count / total_obs)
        summary$MNAR_rate <- as.numeric(mnar_count / total_obs)
      }
      summary
    },
    
    save_outputs = function(df, base_path) {
      base_dir <- dirname(base_path)
      base_name <- basename(base_path)
      if (identical(base_dir, "") || is.na(base_dir)) base_dir <- "."
      
      data_xlsx   <- file.path(base_dir, paste0(base_name, "_data.xlsx"))
      data_csv    <- file.path(base_dir, paste0(base_name, "_data.csv"))
      summary_csv <- file.path(base_dir, paste0(base_name, "_summary.csv"))
      config_json <- file.path(base_dir, paste0(base_name, "_config.json"))
      
      openxlsx::write.xlsx(df, data_xlsx, overwrite = TRUE)
      write.csv(df, data_csv, row.names = FALSE)
      
      summary <- self$summarize(df)
      write.csv(as.data.frame(summary), summary_csv, row.names = FALSE)
      
      config <- list(
        n_per_class = self$n_per_class,
        add_missingness = self$add_missingness,
        base_random_state = self$base_random_state,
        deterministic = TRUE
      )
      jsonlite::write_json(config, config_json, pretty = TRUE, auto_unbox = TRUE)
    },
    
    generate = function(keep_internal_label = FALSE) {
      
      # - On crée 2*n_per_class lignes.
      # - Lyme_true alterne (1,0,1,0...) : ça fabrique exactement n_per_class Lyme + n_per_class non-Lyme.
      
      n_per_class <- self$n_per_class
      add_missingness <- self$add_missingness
      df <- data.frame(Lyme_true = rep(c(1L, 0L), times = n_per_class), stringsAsFactors = FALSE)
      n  <- nrow(df)
      
      idx_lyme <- which(df$Lyme_true == 1L)
      idx_non  <- which(df$Lyme_true == 0L)
      
      # ----------------------------------------------------------------
      # 2) Identité (déterministe)
      # ----------------------------------------------------------------
      # - On invente des chevaux Cheval_0000, Cheval_0001, ...
      # - Age : cycle 4..20
      # - Sexe : 0/1 alterné
      
      df$Nom_du_Cheval <- sprintf("Cheval_%04d", 0:(n - 1))
      df$Age_du_cheval <- as.integer(cycle_to_n(4:20, n))
      df$Sexe <- as.integer(rep(c(0L, 1L), length.out = n))
      
      # ----------------------------------------------------------------
      # 3) Type_de_cheval (3 modalités strictes)
      # ----------------------------------------------------------------
      # On tourne en boucle sur les 3 catégories.
      df$Type_de_cheval <- cycle_to_n(c("Selle_et_poneys", "Trait", "Course"), n)
      
      # ----------------------------------------------------------------
      # 4) Saison (cycle fixe)
      # ----------------------------------------------------------------
      # Hiver, Printemps, Été, Automne, puis on recommence.
      df$Season <- cycle_to_n(c("Hiver", "Printemps", "Été", "Automne"), n)
      
      # ----------------------------------------------------------------
      # 5) Exposition / contexte (sans règles "plus de sortie => plus de tiques")
      # ----------------------------------------------------------------
      # - La fréquence de sortie (0..7) est un simple motif.
      # - Pour les cas Lyme parfaits : on force >=3 (sinon ça ferait bizarre d’avoir Lyme “parfait” sans exposition).
      df$Freq_acces_exterieur_sem <- as.integer(cycle_to_n(0:7, n))
      if (length(idx_lyme) > 0) {
        df$Freq_acces_exterieur_sem[idx_lyme] <- pmax(df$Freq_acces_exterieur_sem[idx_lyme], 3L)
      }
      
      # Extérieur végétalisé :
      # - si freq=0 => "non"
      # - sinon => "oui", sauf 1 ligne sur 9 forcée en "non" (pour varier)
      # - Lyme parfait => toujours "oui"
      df$Exterieur_vegetalisé <- ifelse(df$Freq_acces_exterieur_sem == 0L, "non", "oui")
      idx_force_nonveg <- which(df$Freq_acces_exterieur_sem > 0L & (seq_len(n) %% 9) == 0)
      if (length(idx_force_nonveg) > 0) df$Exterieur_vegetalisé[idx_force_nonveg] <- "non"
      if (length(idx_lyme) > 0) df$Exterieur_vegetalisé[idx_lyme] <- "oui"
      
      # Tiques :
      # - jamais si freq=0 OU extérieur non végétalisé
      # - sinon motif alterné 0/1 (sans dépendre de la fréquence)
      # - Lyme parfait => 1 (tiques observées)
      df$Tiques_semaines_précédentes <- 0L
      tick_pattern <- as.integer((seq_len(n) %% 2) == 0)  # 0,1,0,1...
      idx_can_tick <- which(df$Freq_acces_exterieur_sem > 0L & df$Exterieur_vegetalisé == "oui")
      if (length(idx_can_tick) > 0) df$Tiques_semaines_précédentes[idx_can_tick] <- tick_pattern[idx_can_tick]
      if (length(idx_lyme) > 0) df$Tiques_semaines_précédentes[idx_lyme] <- 1L
      
      # Classe de risque (3 niveaux stricts)
      # - motif fixe sur 3 modalités
      # - Lyme parfait => "fort"
      # - si freq=0 => "Faible ou méconnu" (cohérence)
      base_risk <- cycle_to_n(c("Faible ou méconnu", "intermédiaire", "fort"), n)
      df[["Classe de risque"]] <- base_risk
      if (length(idx_lyme) > 0) df[["Classe de risque"]][idx_lyme] <- "fort"
      idx_freq0 <- which(df$Freq_acces_exterieur_sem == 0L)
      if (length(idx_freq0) > 0) df[["Classe de risque"]][idx_freq0] <- "Faible ou méconnu"
      
      # ----------------------------------------------------------------
      # 6) Examen clinique / exclusions / bilans (cas parfaits : pas de confusion)
      # ----------------------------------------------------------------
      # - Cas parfait Lyme : pas d’autre diagnostic évident (donc bilans “normaux/négatifs”).
      # - Cas parfait non-Lyme : on introduit des anomalies pour “expliquer autrement” (et on met Examen_clinique=0).
      df$Examen_clinique <- 1L
      
      df$piroplasmose_neg <- 1L
      df$ehrlichiose_neg <- 1L
      df$Bilan_sanguin_normal <- 1L
      df$NFS_normale <- 1L
      df$Parametres_musculaires_normaux <- 1L
      df$Parametres_renaux_normaux <- 1L
      df$Parametres_hepatiques_normaux <- 1L
      df$SAA_normal <- 1L
      df$Fibrinogène_normal <- 1L
      
      if (length(idx_non) > 0) {
        nn <- length(idx_non)
        non_rank <- seq_len(nn)
        
        idx_piro_pos <- idx_non[non_rank %% 17 == 0]
        idx_ehrl_pos <- idx_non[non_rank %% 19 == 0]
        
        idx_bilan_abn <- unique(c(idx_piro_pos, idx_ehrl_pos, idx_non[non_rank %% 23 == 0]))
        idx_nfs_abn   <- unique(c(idx_bilan_abn, idx_non[non_rank %% 29 == 0]))
        idx_musc_abn  <- idx_non[non_rank %% 31 == 0]
        idx_ren_abn   <- idx_non[non_rank %% 37 == 0]
        idx_hep_abn   <- idx_non[non_rank %% 41 == 0]
        
        if (length(idx_piro_pos) > 0) df$piroplasmose_neg[idx_piro_pos] <- 0L
        if (length(idx_ehrl_pos) > 0) df$ehrlichiose_neg[idx_ehrl_pos] <- 0L
        
        if (length(idx_bilan_abn) > 0) df$Bilan_sanguin_normal[idx_bilan_abn] <- 0L
        if (length(idx_nfs_abn) > 0) df$NFS_normale[idx_nfs_abn] <- 0L
        if (length(idx_musc_abn) > 0) df$Parametres_musculaires_normaux[idx_musc_abn] <- 0L
        if (length(idx_ren_abn) > 0) df$Parametres_renaux_normaux[idx_ren_abn] <- 0L
        if (length(idx_hep_abn) > 0) df$Parametres_hepatiques_normaux[idx_hep_abn] <- 0L
        
        idx_other_dx <- unique(c(idx_piro_pos, idx_ehrl_pos, idx_bilan_abn, idx_nfs_abn, idx_musc_abn, idx_ren_abn, idx_hep_abn))
        if (length(idx_other_dx) > 0) df$Examen_clinique[idx_other_dx] <- 0L
      }
      
      # ----------------------------------------------------------------
      # 7) Sérologies / PCR sang (cas parfaits)
      # ----------------------------------------------------------------
      # Explication ultra simple :
      # - Ici on force une séparation parfaite :
      #   Lyme_true=1 => sérologies positives
      #   Lyme_true=0 => sérologies négatives
      # - PCR sang : toujours 0 
      df$PCR_sang_pos <- 0L
      
      df$ELISA_pos   <- as.integer(df$Lyme_true == 1L)
      df$WB_pos      <- as.integer(df$Lyme_true == 1L)
      df$SNAP_C6_pos <- as.integer(df$Lyme_true == 1L)
      df$IFAT_pos    <- as.integer(df$Lyme_true == 1L)
      
      df$ELISA_OspA_pos <- 0L
      df$ELISA_OspF_pos <- as.integer(df$Lyme_true == 1L)
      df$ELISA_p39      <- as.integer(df$Lyme_true == 1L)
      
      # ----------------------------------------------------------------
      # 8) Examens ciblés (PCR/LCR/histo) : init à 0 puis activés par scénarios Lyme
      # ----------------------------------------------------------------
      # - On crée les colonnes, toutes à 0.
      # - Ensuite, selon le “scénario Lyme”, on passe certaines à 1.
      targeted_cols <- c(
        "PCR_LCR_pos",
        "PCR_synoviale_pos",
        "PCR_peau_pos",
        "PCR_humeur_aqueuse_pos",
        "PCR_tissu_nerveux_pos",
        "IHC_tissulaire_pos",
        "Coloration_argent_pos",
        "FISH_tissulaire_pos",
        "PCR_liquide_articulaire_pos",
        "LCR_pleiocytose",
        "LCR_proteines_augmentees"
      )
      df <- zero_cols(df, targeted_cols)
      
      # ----------------------------------------------------------------
      # 9) Signes cliniques : init à 0 puis activés par scénarios Lyme
      # ----------------------------------------------------------------
      signes_cols <- c(
        "Meningite",
        "Radiculonevrite",
        "Troubles_de_la_demarche",
        "Dysphagie",
        "Fasciculations_musculaires",
        "Troubles_du_comportement",
        "Hyperesthesie_cutanee",
        "Deficits_nerfs_craniens",
        "Detresse_respiratoire_laryngee",
        "Amyotrophie",
        "Raideur_cervicale",
        
        "Uveite_bilaterale",
        "Cecite_avec_cause_inflammatoire",
        "Synechies",
        "Atrophie",
        "Dyscories",
        "Myosis",
        "Blepharospasme",
        "Epiphora",
        
        "Synovite_avec_epanchement_articulaire",
        "Arthrite",
        
        "Pseudolyphome_cutane",
        "Pododermatite",
        
        "Abattement",
        "Boiterie",
        "Mauvaise_performance",
        "Douleurs_diffuses"
      )
      df <- zero_cols(df, signes_cols)
      
      # Si freq=0 : on “verrouille” à 0 (pas de faisceau Lyme si pas d’exposition)
      if (length(idx_freq0) > 0) {
        for (sc in signes_cols) df[idx_freq0, sc] <- 0L
        for (tc in targeted_cols) df[idx_freq0, tc] <- 0L
      }
      
      # Non-Lyme : quelques signes NON spécifiques (fatigue, perf, douleurs…)
      if (length(idx_non) > 0) {
        nn <- length(idx_non)
        non_rank <- seq_len(nn)
        
        df$Abattement[idx_non[non_rank %% 4 == 0]] <- 1L
        df$Mauvaise_performance[idx_non[non_rank %% 5 == 0]] <- 1L
        df$Douleurs_diffuses[idx_non[non_rank %% 6 == 0]] <- 1L
        df$Boiterie[idx_non[non_rank %% 7 == 0]] <- 1L
      }
      
      # Lyme : signes généraux (cohérent)
      if (length(idx_lyme) > 0) {
        df$Abattement[idx_lyme] <- 1L
        df$Mauvaise_performance[idx_lyme] <- 1L
      }
      
      # ----------------------------------------------------------------
      # 10) IDCV/CVID (neutre)
      # ----------------------------------------------------------------
      df$CVID <- 0L
      df$Hypoglobulinemie <- 0L
      
      # ----------------------------------------------------------------
      # 11) Scénarios Lyme parfaits (déterministes)
      # ----------------------------------------------------------------
      # Parmi les Lyme, on force 4 “tableaux” :
      # - NEURO : neurologique + LCR + PCR LCR + PCR tissu nerveux
      # - UVEITE : uvéite + PCR humeur aqueuse
      # - CUTANE : pseudolymphome/peau + PCR peau
      # - ARTICULAIRE : synovite/arthrite + PCR synovie/liquide
      if (length(idx_lyme) > 0) {
        
        scenario <- cycle_to_n(c("NEURO", "UVEITE", "CUTANE", "ARTICULAIRE"), length(idx_lyme))
        
        idx_NEURO <- idx_lyme[scenario == "NEURO"]
        idx_UVE   <- idx_lyme[scenario == "UVEITE"]
        idx_CUT   <- idx_lyme[scenario == "CUTANE"]
        idx_ART   <- idx_lyme[scenario == "ARTICULAIRE"]
        
        df <- set_if_col(df, "Meningite", idx_NEURO, 1L)
        df <- set_if_col(df, "Radiculonevrite", idx_NEURO, 1L)
        df <- set_if_col(df, "Troubles_de_la_demarche", idx_NEURO, 1L)
        df <- set_if_col(df, "Dysphagie", idx_NEURO, 1L)
        df <- set_if_col(df, "Raideur_cervicale", idx_NEURO, 1L)
        df <- set_if_col(df, "Amyotrophie", idx_NEURO, 1L)
        df <- set_if_col(df, "Hyperesthesie_cutanee", idx_NEURO, 1L)
        
        df <- set_if_col(df, "LCR_pleiocytose", idx_NEURO, 1L)
        df <- set_if_col(df, "LCR_proteines_augmentees", idx_NEURO, 1L)
        df <- set_if_col(df, "PCR_LCR_pos", idx_NEURO, 1L)
        df <- set_if_col(df, "PCR_tissu_nerveux_pos", idx_NEURO, 1L)
        
        df <- set_if_col(df, "Uveite_bilaterale", idx_UVE, 1L)
        df <- set_if_col(df, "Myosis", idx_UVE, 1L)
        df <- set_if_col(df, "Dyscories", idx_UVE, 1L)
        df <- set_if_col(df, "Blepharospasme", idx_UVE, 1L)
        df <- set_if_col(df, "Epiphora", idx_UVE, 1L)
        df <- set_if_col(df, "Synechies", idx_UVE, 1L)
        df <- set_if_col(df, "PCR_humeur_aqueuse_pos", idx_UVE, 1L)
        
        df <- set_if_col(df, "Pseudolyphome_cutane", idx_CUT, 1L)
        df <- set_if_col(df, "Pododermatite", idx_CUT, 1L)
        df <- set_if_col(df, "PCR_peau_pos", idx_CUT, 1L)
        
        df <- set_if_col(df, "Synovite_avec_epanchement_articulaire", idx_ART, 1L)
        df <- set_if_col(df, "Arthrite", idx_ART, 1L)
        df <- set_if_col(df, "Boiterie", idx_ART, 1L)
        df <- set_if_col(df, "PCR_synoviale_pos", idx_ART, 1L)
        df <- set_if_col(df, "PCR_liquide_articulaire_pos", idx_ART, 1L)
      }
      
      # ----------------------------------------------------------------
      # 12) Missingness (0/1/2) selon règles demandées
      # ----------------------------------------------------------------
      # - On crée une colonne “*_missing_code” pour CHAQUE variable exportée.
      # - code 0 : valeur présente
      # - code 1 : manquant type MCAR (oubli “au hasard” simulé)
      # - code 2 : manquant type MNAR (test non réalisé car pas d’indication)
      
      missing_name_map <- list(
        "Nom_du_Cheval" = "Nom_du_Cheval_missing_code",
        "Age_du_cheval" = "Age_du_cheval_missing_code",
        "Sexe" = "Sexe_missing_code",
        "Type_de_cheval" = "Type_de_cheval_missing_code",
        "Season" = "Season_missing_code",
        "Classe de risque" = "Classe_de_risque_missing_code",
        "Exterieur_vegetalisé" = "Exterieur_vegetalisé_missing_code",
        "Freq_acces_exterieur_sem" = "Freq_acces_exterieur_sem_missing_code",
        "Tiques_semaines_précédentes" = "Tiques_semaines_précédentes_missing_code",
        "Examen_clinique" = "Examen_clinique_missing_code",
        
        "piroplasmose_neg" = "piroplasmose_neg_missing_code",
        "ehrlichiose_neg" = "ehrlichiose_neg_missing_code",
        "Bilan_sanguin_normal" = "Bilan_sanguin_normal_missing_code",
        "NFS_normale" = "NFS_normale_missing_code",
        "Parametres_musculaires_normaux" = "Parametres_musculaires_normaux_missing_code",
        "Parametres_renaux_normaux" = "Parametres_renaux_normaux_missing_code",
        "Parametres_hepatiques_normaux" = "Parametres_hepatiques_normaux_missing_code",
        "SAA_normal" = "SAA_normal_missing_code",
        "Fibrinogène_normal" = "Fibrinogène_normal_missing_code",
        
        "ELISA_pos" = "ELISA_pos_missing_code",
        "ELISA_OspA_pos" = "ELISA_OspA_pos_missing_code",
        "ELISA_OspF_pos" = "ELISA_OspF_pos_missing_code",
        "ELISA_p39" = "ELISA_p39_missing_code",
        "WB_pos" = "WB_pos_missing_code",
        "PCR_sang_pos" = "PCR_sang_pos_missing_code",
        "SNAP_C6_pos" = "SNAP_C6_pos_missing_code",
        "IFAT_pos" = "IFAT_pos_missing_code",
        
        "PCR_LCR_pos" = "PCR_LCR_pos_missing_code",
        "PCR_synoviale_pos" = "PCR_synoviale_pos_missing_code",
        "PCR_peau_pos" = "PCR_peau_pos_missing_code",
        "PCR_humeur_aqueuse_pos" = "PCR_humeur_aqueuse_pos_missing_code",
        "PCR_tissu_nerveux_pos" = "PCR_tissu_nerveux_pos_missing_code",
        "IHC_tissulaire_pos" = "IHC_tissulaire_pos_missing_code",
        "Coloration_argent_pos" = "Coloration_argent_pos_missing_code",
        "FISH_tissulaire_pos" = "FISH_tissulaire_pos_missing_code",
        "PCR_liquide_articulaire_pos" = "PCR_liquide_articulaire_pos_missing_code",
        "LCR_pleiocytose" = "LCR_pleiocytose_missing_code",
        "LCR_proteines_augmentees" = "LCR_proteines_augmentees_missing_code",
        "CVID" = "CVID_missing_code",
        "Hypoglobulinemie" = "Hypoglobulinemie_missing_code",
        
        "Meningite" = "Meningite_missing_code",
        "Radiculonevrite" = "Radiculonevrite_missing_code",
        "Troubles_de_la_demarche" = "Troubles_de_la_demarche_missing_code",
        "Dysphagie" = "Dysphagie_missing_code",
        "Fasciculations_musculaires" = "Fasciculations_musculaires_missing_code",
        "Troubles_du_comportement" = "Troubles_du_comportement_missing_code",
        "Hyperesthesie_cutanee" = "Hyperesthesie_cutanee_missing_code",
        "Deficits_nerfs_craniens" = "Deficits_nerfs_craniens_missing_code",
        "Detresse_respiratoire_laryngee" = "Detresse_respiratoire_laryngee_missing_code",
        "Amyotrophie" = "Amyotrophie_missing_code",
        "Raideur_cervicale" = "Raideur_cervicale_missing_code",
        
        "Uveite_bilaterale" = "Uveite_bilaterale_missing_code",
        "Cecite_avec_cause_inflammatoire" = "Cecite_avec_cause_inflammatoire_missing_code",
        "Synechies" = "Synechies_missing_code",
        "Atrophie" = "Atrophie_missing_code",
        "Dyscories" = "Dyscories_missing_code",
        "Myosis" = "Myosis_missing_code",
        "Blepharospasme" = "Blepharospasme_missing_code",
        "Epiphora" = "Epiphora_missing_code",
        
        "Synovite_avec_epanchement_articulaire" = "Synovite_avec_epanchement_articulaire_missing_code",
        "Arthrite" = "Arthrite_missing_code",
        "Pseudolyphome_cutane" = "Pseudolyphome_cutane_missing_code",
        "Pododermatite" = "Pododermatite_missing_code",
        
        "Abattement" = "Abattement_missing_code",
        "Boiterie" = "Boiterie_missing_code",
        "Mauvaise_performance" = "Mauvaise_performance_missing_code",
        "Douleurs_diffuses" = "Douleurs_diffuses_missing_code"
      )
      
      if (add_missingness) {
        
        # 12.1) Créer toutes les colonnes *_missing_code à 0
        exported_cols <- names(missing_name_map)
        for (c in exported_cols) {
          mc <- missing_name_map[[c]]
          df[[mc]] <- 0L
        }
        
        # 12.2) liste des analyses (MNAR), utile pour lecture humaine
        analysis_cols <- c(
          "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
          "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
          "SAA_normal","Fibrinogène_normal",
          "ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","WB_pos","PCR_sang_pos","SNAP_C6_pos","IFAT_pos",
          "PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos",
          "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos","PCR_liquide_articulaire_pos",
          "LCR_pleiocytose","LCR_proteines_augmentees","CVID","Hypoglobulinemie"
        )
        
        # 12.3) MCAR sur contexte + signes
        df <- apply_mcar_pattern(df, "Age_du_cheval", missing_name_map[["Age_du_cheval"]], every_k = 50)
        df <- apply_mcar_pattern(df, "Type_de_cheval", missing_name_map[["Type_de_cheval"]], every_k = 80)
        df <- apply_mcar_pattern(df, "Season", missing_name_map[["Season"]], every_k = 90)
        df <- apply_mcar_pattern(df, "Exterieur_vegetalisé", missing_name_map[["Exterieur_vegetalisé"]], every_k = 100)
        df <- apply_mcar_pattern(df, "Freq_acces_exterieur_sem", missing_name_map[["Freq_acces_exterieur_sem"]], every_k = 110)
        df <- apply_mcar_pattern(df, "Tiques_semaines_précédentes", missing_name_map[["Tiques_semaines_précédentes"]], every_k = 120)
        df <- apply_mcar_pattern(df, "Classe de risque", missing_name_map[["Classe de risque"]], every_k = 130)
        
        for (sc in signes_cols) {
          df <- apply_mcar_pattern(df, sc, missing_name_map[[sc]], every_k = 75)
        }
        
        # 12.4) MNAR pour les analyses chez non-Lyme uniquement
        idx_non_now <- which(df$Lyme_true == 0L)
        
        idx_non_only <- function(mask) {
          which(mask & df$Lyme_true == 0L)
        }
        
        no_ocular <- (df$Uveite_bilaterale == 0 | is.na(df$Uveite_bilaterale)) &
          (df$Cecite_avec_cause_inflammatoire == 0 | is.na(df$Cecite_avec_cause_inflammatoire))
        idx_no_ocular <- idx_non_only(no_ocular)
        df <- apply_mnar_deterministic(df, "PCR_humeur_aqueuse_pos", missing_name_map[["PCR_humeur_aqueuse_pos"]],
                                       idx_no_ocular, keep_every_k = 5)
        
        neuro_any <- (df$Meningite == 1) | (df$Radiculonevrite == 1) | (df$Troubles_de_la_demarche == 1) |
          (df$Dysphagie == 1) | (df$Hyperesthesie_cutanee == 1) | (df$Deficits_nerfs_craniens == 1) |
          (df$Detresse_respiratoire_laryngee == 1)
        neuro_any[is.na(neuro_any)] <- FALSE
        idx_no_neuro <- idx_non_only(!neuro_any)
        
        df <- apply_mnar_deterministic(df, "PCR_LCR_pos", missing_name_map[["PCR_LCR_pos"]], idx_no_neuro, keep_every_k = 5)
        df <- apply_mnar_deterministic(df, "LCR_pleiocytose", missing_name_map[["LCR_pleiocytose"]], idx_no_neuro, keep_every_k = 5)
        df <- apply_mnar_deterministic(df, "LCR_proteines_augmentees", missing_name_map[["LCR_proteines_augmentees"]], idx_no_neuro, keep_every_k = 5)
        df <- apply_mnar_deterministic(df, "PCR_tissu_nerveux_pos", missing_name_map[["PCR_tissu_nerveux_pos"]], idx_no_neuro, keep_every_k = 6)
        
        art_any <- (df$Synovite_avec_epanchement_articulaire == 1) | (df$Arthrite == 1)
        art_any[is.na(art_any)] <- FALSE
        idx_no_art <- idx_non_only(!art_any)
        
        df <- apply_mnar_deterministic(df, "PCR_synoviale_pos", missing_name_map[["PCR_synoviale_pos"]], idx_no_art, keep_every_k = 5)
        df <- apply_mnar_deterministic(df, "PCR_liquide_articulaire_pos", missing_name_map[["PCR_liquide_articulaire_pos"]], idx_no_art, keep_every_k = 5)
        
        cut_any <- (df$Pseudolyphome_cutane == 1) | (df$Pododermatite == 1)
        cut_any[is.na(cut_any)] <- FALSE
        idx_no_cut <- idx_non_only(!cut_any)
        df <- apply_mnar_deterministic(df, "PCR_peau_pos", missing_name_map[["PCR_peau_pos"]], idx_no_cut, keep_every_k = 5)
        
        idx_hist <- idx_non_now
        df <- apply_mnar_deterministic(df, "IHC_tissulaire_pos", missing_name_map[["IHC_tissulaire_pos"]], idx_hist, keep_every_k = 8)
        df <- apply_mnar_deterministic(df, "Coloration_argent_pos", missing_name_map[["Coloration_argent_pos"]], idx_hist, keep_every_k = 8)
        df <- apply_mnar_deterministic(df, "FISH_tissulaire_pos", missing_name_map[["FISH_tissulaire_pos"]], idx_hist, keep_every_k = 8)
        
        strong_any <- (df$Meningite == 1) | (df$Radiculonevrite == 1) | (df$Troubles_de_la_demarche == 1) |
          (df$Uveite_bilaterale == 1) | (df$Synovite_avec_epanchement_articulaire == 1) | (df$Pseudolyphome_cutane == 1)
        strong_any[is.na(strong_any)] <- FALSE
        
        idx_low_suspicion <- idx_non_only(!strong_any)
        
        sero_cols <- c("ELISA_pos","WB_pos","SNAP_C6_pos","IFAT_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","PCR_sang_pos")
        for (cc in sero_cols) {
          df <- apply_mnar_deterministic(df, cc, missing_name_map[[cc]], idx_low_suspicion, keep_every_k = 4)
        }
        
        exclusion_cols <- c(
          "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
          "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
          "SAA_normal","Fibrinogène_normal"
        )
        for (cc in exclusion_cols) {
          df <- apply_mnar_deterministic(df, cc, missing_name_map[[cc]], idx_low_suspicion, keep_every_k = 4)
        }
        
        df <- apply_mnar_deterministic(df, "CVID", missing_name_map[["CVID"]], idx_low_suspicion, keep_every_k = 6)
        df <- apply_mnar_deterministic(df, "Hypoglobulinemie", missing_name_map[["Hypoglobulinemie"]], idx_low_suspicion, keep_every_k = 6)
      }
      
      # ----------------------------------------------------------------
      # 13) Mise en ordre + sortie
      # ----------------------------------------------------------------
      # - On choisit un ordre “propre” des colonnes
      # - Puis on renvoie le df final, avec ou sans Lyme_true (suivant keep_internal_label)
      
      ordered_main_cols <- c(
        "Nom_du_Cheval","Age_du_cheval","Sexe","Type_de_cheval","Season","Classe de risque",
        "Exterieur_vegetalisé","Freq_acces_exterieur_sem","Tiques_semaines_précédentes","Examen_clinique",
        "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
        "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
        "SAA_normal","Fibrinogène_normal","ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39",
        "WB_pos","PCR_sang_pos","SNAP_C6_pos","IFAT_pos","PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos",
        "PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos","IHC_tissulaire_pos","Coloration_argent_pos",
        "FISH_tissulaire_pos","PCR_liquide_articulaire_pos","LCR_pleiocytose","LCR_proteines_augmentees",
        "CVID","Hypoglobulinemie",
        "Meningite","Radiculonevrite","Troubles_de_la_demarche","Dysphagie",
        "Fasciculations_musculaires","Troubles_du_comportement","Hyperesthesie_cutanee",
        "Deficits_nerfs_craniens","Detresse_respiratoire_laryngee","Amyotrophie","Raideur_cervicale",
        "Uveite_bilaterale","Cecite_avec_cause_inflammatoire","Synechies","Atrophie","Dyscories","Myosis",
        "Blepharospasme","Epiphora",
        "Synovite_avec_epanchement_articulaire","Arthrite","Pseudolyphome_cutane","Pododermatite",
        "Abattement","Boiterie","Mauvaise_performance","Douleurs_diffuses"
      )
      
      ordered_missing_cols <- unname(unlist(missing_name_map))
      ordered_missing_cols <- ordered_missing_cols[!is.na(ordered_missing_cols)]
      
      final_cols <- intersect(ordered_main_cols, names(df))
      final_cols <- c(final_cols, intersect(ordered_missing_cols, names(df)))
      
      if (keep_internal_label) {
        out <- df[, c("Lyme_true", final_cols), drop = FALSE]
      } else {
        out <- df[, final_cols, drop = FALSE]
      }
      
      out
    }
  )
)


# --------------------------------------------------------------------------
# Bloc 4 — Wrapper simple
# --------------------------------------------------------------------------
# - Petite fonction “raccourci” : tu appelles ça, ça te sort un dataset.
generate_equine_lyme_perfect_dataset <- function(n_per_class = 100, random_state = 42, add_missingness = TRUE,
                                                 keep_internal_label = FALSE) {
  gen <- EquineLymePerfectDatasetGenerator$new(
    n_per_class = n_per_class,
    base_random_state = random_state,
    add_missingness = add_missingness
  )
  gen$generate(keep_internal_label = keep_internal_label)
}


# --------------------------------------------------------------------------
# Bloc 5 — Exemple d’utilisation
# --------------------------------------------------------------------------
# - Si tu lances le script “tout seul”, ce bloc s’exécute.
# - Il génère 1000 Lyme + 1000 non-Lyme, affiche un aperçu, et exporte.
if (sys.nframe() == 0) {
  
  gen <- EquineLymePerfectDatasetGenerator$new(
    n_per_class = 50,
    base_random_state = 123,   # non utilisé (déterministe) mais conservé
    add_missingness = TRUE     # MNAR analyses / MCAR reste
  )
  
  df_demo <- gen$generate(keep_internal_label = TRUE)
  print(head(df_demo))
  cat("Shape:", nrow(df_demo), "x", ncol(df_demo), "\n")
  
  base_path <- "C:/Users/q.lamboley/Downloads/jeu_fictif_lyme_equine_cas_parfaits"
  gen$save_outputs(df_demo, base_path)
  
  openxlsx::write.xlsx(
    df_demo,
    "C:/Users/q.lamboley/Downloads/jeu_fictif_lyme_equine_cas_parfaits.xlsx",
    overwrite = TRUE
  )
}










































































































# ============================================================
# CatBoost (classification) + validation croisée 5 folds
# Remplace le Random Forest ranger du bloc précédent
# ============================================================
# Explication ultra simple :
# - CatBoost est un gradient boosting sur arbres, très performant.
# - Gros avantage : il gère NATIVEMENT les NA (valeurs manquantes) :
#   pas besoin de les remplacer/imputer pour l'entraînement.
# - Il gère aussi très bien les variables catégorielles.
# - On fait une CV 5 folds, puis on entraîne un modèle final sur tout le dataset.
# - Ensuite on simule des "nouveaux chevaux" imparfaits et on prédit P(Lyme).
#
# IMPORTANT :
# - On retire Nom_du_Cheval pour éviter que le modèle apprenne des patterns d'ID.
# - On garde les colonnes *_missing_code : elles sont informatives (et réalistes)
#   car elles indiquent POURQUOI une donnée manque (MCAR/MNAR).

# -----------------------------
# 0) Dépendances
# -----------------------------
suppressWarnings({
  pkgs <- c("catboost", "dplyr", "rsample", "pROC", "tibble", "openxlsx")
  for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
})
library(catboost)
library(dplyr)
library(rsample)
library(pROC)
library(tibble)
library(openxlsx)
library(stringr)

# # ============================================================
# # INSTALL CATBOOST (Windows) - depuis le binaire GitHub
# # ============================================================
# 
# # 0) Outils d'installation
# if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
# 
# # 1) Fonction utilitaire : détecte ta version de R (ex: "4.4")
# r_minor <- paste0(R.version$major, ".", R.version$minor)
# 
# cat("R version détectée :", r_minor, "\n")
# 
# # 2) Essai d'installation automatique depuis les releases CatBoost
# #    -> CatBoost publie des .zip "catboost-R-Windows-<Rver>.zip"
# #    -> On essaie d'abord avec ta version, sinon on essaie des alternatives proches.
# install_catboost_from_github <- function(candidate_r_versions = c(r_minor, "4.4", "4.3", "4.2", "4.1")) {
#   ok <- FALSE
#   last_err <- NULL
#   
#   for (rv in unique(candidate_r_versions)) {
#     url <- paste0(
#       "https://github.com/catboost/catboost/releases/latest/download/",
#       "catboost-R-Windows-", rv, ".zip"
#     )
#     cat("\nTentative :", url, "\n")
#     
#     res <- tryCatch({
#       remotes::install_url(url, upgrade = "never", dependencies = FALSE, quiet = TRUE)
#       TRUE
#     }, error = function(e) {
#       last_err <<- e$message
#       FALSE
#     })
#     
#     if (isTRUE(res)) { ok <- TRUE; break }
#   }
#   
#   if (!ok) {
#     stop("Impossible d'installer CatBoost depuis les binaires GitHub. Dernière erreur: ", last_err)
#   }
# }
# 
# install_catboost_from_github()
# 
# # 3) Test
# library(catboost)
# cat("CatBoost installé ✅\n")
# 
# 
# 
# 
# 
# # ============================================================
# # Installation CatBoost (R) sur Windows via GitHub Releases
# # - NE PAS utiliser install.packages("catboost") : pas sur CRAN
# # - Télécharge l'asset officiel: catboost-R-windows-x86_64-<ver>.tgz
# # ============================================================
# 
# install_catboost_windows <- function(
#     repo = "catboost/catboost",
#     prefer_asset_regex = "^catboost-R-windows-x86_64-.*\\.tgz$",
#     verbose = TRUE
# ) {
#   # Dépendance légère pour parser le JSON
#   if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
#   
#   api <- paste0("https://api.github.com/repos/", repo, "/releases/latest")
#   if (verbose) cat("API GitHub:", api, "\n")
#   
#   rel <- jsonlite::fromJSON(api)
#   
#   if (is.null(rel$assets) || nrow(rel$assets) == 0) {
#     stop("Aucun asset trouvé dans la release GitHub 'latest'.")
#   }
#   
#   assets <- rel$assets
#   idx <- grep(prefer_asset_regex, assets$name)
#   
#   if (length(idx) == 0) {
#     stop(
#       "Je ne trouve pas d'asset correspondant à: ", prefer_asset_regex, "\n",
#       "Assets disponibles:\n- ", paste(assets$name, collapse = "\n- ")
#     )
#   }
#   
#   # Si plusieurs matches, on prend le 1er
#   asset_name <- assets$name[idx[1]]
#   url <- assets$browser_download_url[idx[1]]
#   
#   if (verbose) {
#     cat("Release:", rel$tag_name, "\n")
#     cat("Asset choisi:", asset_name, "\n")
#     cat("URL:", url, "\n")
#   }
#   
#   dest <- file.path(tempdir(), asset_name)
#   if (verbose) cat("Téléchargement vers:", dest, "\n")
#   
#   # download.file avec mode binaire (Windows)
#   utils::download.file(url, destfile = dest, mode = "wb", quiet = !verbose)
#   
#   if (!file.exists(dest) || file.info(dest)$size < 10000) {
#     stop("Téléchargement incomplet ou fichier trop petit. Vérifie proxy/antivirus/réseau.")
#   }
#   
#   if (verbose) cat("Installation R package depuis:", dest, "\n")
#   install.packages(dest, repos = NULL, type = "source")
#   
#   if (verbose) cat("Test: library(catboost)\n")
#   suppressPackageStartupMessages(library(catboost))
#   
#   invisible(TRUE)
# }
# 
# # --- Lance l'installation
# install_catboost_windows()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 























# -----------------------------
# 1) Récupérer / générer les données
# -----------------------------
if (exists("df_demo")) {
  df <- df_demo
} else if (exists("generate_equine_lyme_perfect_dataset")) {
  df <- generate_equine_lyme_perfect_dataset(
    n_per_class = 50,
    random_state = 123,
    add_missingness = TRUE,
    keep_internal_label = TRUE
  )
} else {
  stop("Je ne trouve ni 'df_demo' ni la fonction 'generate_equine_lyme_perfect_dataset()'. Exécute d'abord ton script générateur.")
}

if (!("Lyme_true" %in% names(df))) stop("La colonne cible 'Lyme_true' est absente. Génère avec keep_internal_label = TRUE.")

# -----------------------------
# 2) Pré-traitements (CatBoost-friendly)
# -----------------------------
# - On enlève Nom_du_Cheval (risque de fuite d'info).
# - On crée une cible binaire 0/1 (catboost aime bien).
# - On laisse les NA tels quels (CatBoost les gère).
# - IMPORTANT CatBoost R :
#   * les catégorielles doivent être en factor (sinon cat_features ignoré)
#   * les numériques doivent être en numeric (double), pas integer (évite REAL()/integer)
# - On ajoute un row_id stable pour recoller les OOF sans bidouille de rownames().
df <- df %>%
  dplyr::select(-any_of("Nom_du_Cheval")) %>%
  mutate(
    Lyme_y = as.integer(Lyme_true == 1L),
    row_id = dplyr::row_number()
  )

# Liste des features = tout sauf la cible (et surtout PAS row_id, sinon fuite d'info)
feature_cols <- setdiff(names(df), c("Lyme_true", "Lyme_y", "row_id"))

# --- FIX demandé : empêcher "Sexe" d'être un facteur dominant -> on le retire des features (comme Nom_du_Cheval)
feature_cols <- setdiff(feature_cols, "Sexe")

# Détecter les colonnes catégorielles (character OU factor) puis les forcer en factor
cat_cols <- feature_cols[sapply(df[, feature_cols, drop = FALSE], function(x) is.character(x) || is.factor(x))]
df <- df %>%
  mutate(across(all_of(cat_cols), ~ as.factor(.x)))

# Forcer toutes les colonnes non-catégorielles en numeric (double) (évite REAL()/integer)
num_cols <- setdiff(feature_cols, cat_cols)
df <- df %>%
  mutate(across(all_of(num_cols), ~ as.numeric(.x)))

# Indices (0-based) des variables catégorielles pour CatBoost
cat_features_idx <- which(feature_cols %in% cat_cols) - 1L  # CatBoost R = indices 0-based

# -----------------------------
# 3) Helper : construire un pool CatBoost
# -----------------------------
.make_pool <- function(df_sub, feature_cols, label_col = "Lyme_y", cat_features_idx = integer(0)) {
  X <- df_sub[, feature_cols, drop = FALSE]
  y <- if (!is.null(label_col) && label_col %in% names(df_sub)) df_sub[[label_col]] else NULL
  
  # data.frame => CatBoost utilise les types (factor) et ignore cat_features
  if (is.data.frame(X)) {
    if (is.null(y)) catboost.load_pool(data = X) else catboost.load_pool(data = X, label = y)
  } else {
    if (is.null(y)) catboost.load_pool(data = X, cat_features = cat_features_idx)
    else            catboost.load_pool(data = X, label = y, cat_features = cat_features_idx)
  }
}

# -----------------------------
# 4) Validation croisée 5 folds (stratifiée)
# -----------------------------
set.seed(20260108)
folds <- vfold_cv(df, v = 5, strata = Lyme_y)

# Paramètres CatBoost (robustes par défaut)
# - loss_function = Logloss pour proba
# - eval_metric = AUC
# - iterations élevé + early stopping pour ne pas overfit
# - depth 6-8 souvent bon
# - l2_leaf_reg régularisation
params <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  iterations = 4000,
  learning_rate = 0.05,
  depth = 8,
  l2_leaf_reg = 6,
  random_seed = 20260108,
  # overfitting detector
  od_type = "Iter",
  od_wait = 150,
  # pas de logs trop verbeux
  logging_level = "Silent",
  # meilleure proba
  # (CatBoost est souvent déjà bien calibré, mais ça aide parfois)
  # leaf_estimation_iterations = 10
  allow_writing_files = FALSE
)

fold_metrics <- vector("list", length = 5)
names(fold_metrics) <- paste0("Fold_", seq_len(5))

# On conserve aussi les probabilités out-of-fold pour tracer/choisir seuils si besoin
oof_pred <- rep(NA_real_, nrow(df))

for (i in seq_len(5)) {
  cat("\n====================\n")
  cat("Fold", i, "/ 5\n")
  cat("====================\n")
  
  split_i <- folds$splits[[i]]
  train_raw <- analysis(split_i)
  test_raw  <- assessment(split_i)
  
  pool_train <- .make_pool(train_raw, feature_cols, "Lyme_y", cat_features_idx)
  pool_test  <- .make_pool(test_raw,  feature_cols, "Lyme_y", cat_features_idx)
  
  model_i <- catboost.train(
    learn_pool = pool_train,
    test_pool  = pool_test,
    params = params
  )
  
  # proba classe 1 (= Lyme)
  prob_lyme <- catboost.predict(model_i, pool_test, prediction_type = "Probability")
  
  # stocker OOF proba (utile pour calibration + seuils) — CORRIGÉ via row_id
  oof_pred[test_raw$row_id] <- prob_lyme
  
  pred_class <- ifelse(prob_lyme >= 0.5, 1L, 0L)
  y_true <- test_raw$Lyme_y
  
  TP <- sum(pred_class == 1L & y_true == 1L)
  TN <- sum(pred_class == 0L & y_true == 0L)
  FP <- sum(pred_class == 1L & y_true == 0L)
  FN <- sum(pred_class == 0L & y_true == 1L)
  
  acc  <- (TP + TN) / (TP + TN + FP + FN)
  sens <- if ((TP + FN) == 0) NA else TP / (TP + FN)
  spec <- if ((TN + FP) == 0) NA else TN / (TN + FP)
  
  roc_obj <- tryCatch(
    pROC::roc(response = y_true, predictor = prob_lyme, levels = c(0, 1), quiet = TRUE),
    error = function(e) NULL
  )
  auc <- if (is.null(roc_obj)) NA else as.numeric(pROC::auc(roc_obj))
  
  fold_metrics[[i]] <- tibble(
    fold = i,
    n_train = nrow(train_raw),
    n_test = nrow(test_raw),
    accuracy = acc,
    sensitivity = sens,
    specificity = spec,
    auc = auc,
    TP = TP, TN = TN, FP = FP, FN = FN
  )
  
  print(fold_metrics[[i]])
}

metrics_df <- bind_rows(fold_metrics)

cat("\n====================\n")
cat("Résumé CV (5 folds)\n")
cat("====================\n")
print(metrics_df)

summary_df <- metrics_df %>%
  summarise(
    accuracy_mean = mean(accuracy, na.rm = TRUE),
    accuracy_sd   = sd(accuracy, na.rm = TRUE),
    sensitivity_mean = mean(sensitivity, na.rm = TRUE),
    sensitivity_sd   = sd(sensitivity, na.rm = TRUE),
    specificity_mean = mean(specificity, na.rm = TRUE),
    specificity_sd   = sd(specificity, na.rm = TRUE),
    auc_mean = mean(auc, na.rm = TRUE),
    auc_sd   = sd(auc, na.rm = TRUE)
  )

cat("\n====================\n")
cat("Moyenne ± SD\n")
cat("====================\n")
print(summary_df)


# -----------------------------
# 4bis) Fix robuste des rownames (important pour OOF + prédictions)
# -----------------------------
# rsample conserve les rownames ; on les force pour être sûr
# (utile si tu veux recalculer des seuils sur oof_pred après coup)
if (any(is.na(oof_pred))) {
  cat("\n[INFO] Certaines proba OOF sont NA -> on reconstruit en forçant les rownames.\n")
  df <- df %>% mutate(.row_id = seq_len(n()))
  rownames(df) <- as.character(df$.row_id)
  df$.row_id <- NULL
  # On peut relancer la CV si tu veux OOF complet (optionnel).
}

# -----------------------------
# 5) Entraîner un modèle final sur tout le dataset + importance variables
# -----------------------------
pool_all <- .make_pool(df, feature_cols, "Lyme_y", cat_features_idx)

cat_model_final <- catboost.train(
  learn_pool = pool_all,
  params = params
)

# Importance : PredictionValuesChange est souvent une bonne option.
# (plus robuste que "gain" selon contextes)
imp <- catboost.get_feature_importance(
  cat_model_final,
  pool = pool_all,
  type = "PredictionValuesChange"
)

varimp_df <- tibble(
  variable = feature_cols,
  importance = as.numeric(imp)
) %>%
  arrange(desc(importance))

cat("\n====================\n")
cat("Top 20 importances (modèle final CatBoost)\n")
cat("====================\n")
print(head(varimp_df, 20))

# -----------------------------
# 6) Export (optionnel)
# -----------------------------
out_xlsx <- "C:/Users/q.lamboley/Downloads/CatBoost_CV5_equine_lyme_results.xlsx"
wb <- createWorkbook()

addWorksheet(wb, "CV_folds")
writeData(wb, "CV_folds", metrics_df)

addWorksheet(wb, "CV_summary")
writeData(wb, "CV_summary", summary_df)

addWorksheet(wb, "VarImp_final")
writeData(wb, "VarImp_final", varimp_df)

saveWorkbook(wb, out_xlsx, overwrite = TRUE)
cat("\nRésultats exportés :", out_xlsx, "\n")

# ============================================================
# 7) Générer de nouveaux chevaux "imparfaits" réalistes
# + prédire P(Lyme) avec CatBoost
# ============================================================
# Objectif :
# - Simuler des cas cliniques mal renseignés (NA + incohérences possibles),
#   MAIS en restant cohérent avec la "grammaire" du dataset :
#   * mêmes colonnes
#   * mêmes modalités
#   * mêmes types (0/1, entiers, catégories)
#   * usage des *_missing_code (0/1/2)
#
# NOTE :
# - CatBoost accepte les NA => pas besoin d'imputer x_new.
# - Par contre, on doit garantir les modalités (catégorielles) dans un format compatible.

.pick_or_na <- function(values, p_na = 0.6) {
  if (runif(1) < p_na) return(NA)
  sample(values, size = 1)
}

.sample_from_df_col <- function(df_ref, col, p_na = 0.6, allow_new_level = FALSE) {
  if (!(col %in% names(df_ref))) return(NA)
  v <- df_ref[[col]]
  # On prélève une valeur existante (réaliste), en ignorant NA
  vv <- v[!is.na(v)]
  if (length(vv) == 0) return(NA)
  if (runif(1) < p_na) return(NA)
  out <- sample(vv, 1)
  # Optionnel : créer de très rares valeurs inattendues (par défaut non)
  if (allow_new_level && is.character(v) && runif(1) < 0.01) out <- paste0(out, "_X")
  out
}

# --- FIX : aligner les types du "new_df" sur ceux du train (factors + numeric double)
.coerce_like_train <- function(new_df, df_train, feature_cols, cat_cols) {
  num_cols <- setdiff(feature_cols, cat_cols)
  
  # Catégorielles: factor avec niveaux identiques
  for (cc in cat_cols) {
    if (cc %in% names(new_df)) {
      lv <- levels(df_train[[cc]])
      new_df[[cc]] <- as.character(new_df[[cc]])
      new_df[[cc]] <- factor(new_df[[cc]], levels = lv)
    }
  }
  
  # Numériques: toujours numeric (double)
  for (cc in num_cols) {
    if (cc %in% names(new_df)) {
      new_df[[cc]] <- as.numeric(new_df[[cc]])
    }
  }
  
  new_df
}

.generate_one_realistic_horse <- function(df_ref, feature_cols, p_na_base = 0.55, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # 1 ligne vide avec les mêmes colonnes features
  x_new <- df_ref[0, feature_cols, drop = FALSE]
  x_new[1, ] <- NA
  
  # --- Contexte / identité / exposition
  if ("Age_du_cheval" %in% names(x_new)) x_new$Age_du_cheval <- .pick_or_na(4:25, p_na = p_na_base)
  if ("Sexe" %in% names(x_new))          x_new$Sexe          <- .pick_or_na(c(0L, 1L), p_na = 0.25)
  if ("Type_de_cheval" %in% names(x_new)) x_new$Type_de_cheval <- .pick_or_na(c("Selle_et_poneys","Trait","Course"), p_na = 0.30)
  if ("Season" %in% names(x_new))         x_new$Season         <- .pick_or_na(c("Hiver","Printemps","Été","Automne"), p_na = 0.35)
  
  if ("Classe de risque" %in% names(x_new)) {
    x_new[["Classe de risque"]] <- .pick_or_na(c("Faible ou méconnu","intermédiaire","fort"), p_na = 0.35)
  }
  
  if ("Freq_acces_exterieur_sem" %in% names(x_new)) x_new$Freq_acces_exterieur_sem <- .pick_or_na(0:7, p_na = 0.30)
  if ("Exterieur_vegetalisé" %in% names(x_new))     x_new$Exterieur_vegetalisé     <- .pick_or_na(c("oui","non"), p_na = 0.30)
  if ("Tiques_semaines_précédentes" %in% names(x_new)) x_new$Tiques_semaines_précédentes <- .pick_or_na(c(0L, 1L), p_na = 0.55)
  
  # Cohérence douce : si freq=0 alors tiques=0 (souvent), extérieur non végétalisé (souvent)
  if ("Freq_acces_exterieur_sem" %in% names(x_new)) {
    if (!is.na(x_new$Freq_acces_exterieur_sem) && x_new$Freq_acces_exterieur_sem == 0L) {
      if ("Tiques_semaines_précédentes" %in% names(x_new) && runif(1) < 0.8) x_new$Tiques_semaines_précédentes <- 0L
      if ("Exterieur_vegetalisé" %in% names(x_new) && runif(1) < 0.8) x_new$Exterieur_vegetalisé <- "non"
      if ("Classe de risque" %in% names(x_new) && runif(1) < 0.8) x_new[["Classe de risque"]] <- "Faible ou méconnu"
    }
  }
  
  # --- Examen clinique / bilans (très souvent manquants en vrai)
  if ("Examen_clinique" %in% names(x_new)) x_new$Examen_clinique <- .pick_or_na(c(0L, 1L), p_na = 0.55)
  
  for (cc in c("piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
               "Parametres_musculaires_normaux","Parametres_renaux_normaux",
               "Parametres_hepatiques_normaux","SAA_normal","Fibrinogène_normal")) {
    if (cc %in% names(x_new)) x_new[[cc]] <- .pick_or_na(c(0L, 1L), p_na = 0.70)
  }
  
  # --- Sérologies / PCR sang : souvent MNAR (non fait si faible suspicion)
  for (cc in c("ELISA_pos","WB_pos","SNAP_C6_pos","IFAT_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","PCR_sang_pos")) {
    if (cc %in% names(x_new)) x_new[[cc]] <- .pick_or_na(c(0L, 1L), p_na = 0.75)
  }
  
  # --- Signes : quelques signes généraux + 1 tableau fort possible (rare)
  general_signs <- c("Abattement","Mauvaise_performance","Douleurs_diffuses","Boiterie")
  strong_signs  <- c("Meningite","Troubles_de_la_demarche","Dysphagie","Uveite_bilaterale",
                     "Synovite_avec_epanchement_articulaire","Pseudolyphome_cutane")
  
  for (cc in general_signs) if (cc %in% names(x_new)) x_new[[cc]] <- .pick_or_na(c(0L, 1L), p_na = 0.55)
  for (cc in strong_signs)  if (cc %in% names(x_new)) x_new[[cc]] <- .pick_or_na(c(0L, 1L), p_na = 0.70)
  
  # Les autres signes : on pioche "réaliste" depuis la distribution du dataset (souvent 0)
  other_signs <- setdiff(
    grep("_missing_code$", names(x_new), invert = TRUE, value = TRUE),
    c(feature_cols[feature_cols %in% c("Lyme_true","Lyme_y")], general_signs, strong_signs,
      "Age_du_cheval","Sexe","Type_de_cheval","Season","Classe de risque",
      "Exterieur_vegetalisé","Freq_acces_exterieur_sem","Tiques_semaines_précédentes",
      "Examen_clinique",
      "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
      "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux","SAA_normal","Fibrinogène_normal",
      "ELISA_pos","WB_pos","SNAP_C6_pos","IFAT_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","PCR_sang_pos")
  )
  
  # On limite : on ne touche qu'aux colonnes binaires qu'on reconnaît
  for (cc in other_signs) {
    if (cc %in% names(x_new)) {
      vref <- df_ref[[cc]]
      if (is.numeric(vref) || is.integer(vref)) {
        # Si c'est binaire 0/1 dans le dataset, on respecte
        uniq <- sort(unique(vref[!is.na(vref)]))
        if (length(uniq) <= 3 && all(uniq %in% c(0,1))) {
          x_new[[cc]] <- .pick_or_na(c(0L, 1L), p_na = 0.85)
        }
      }
    }
  }
  
  # --- Missing codes : cohérents (0 = présent, 1 = MCAR, 2 = MNAR)
  miss_cols <- names(x_new)[grepl("_missing_code$", names(x_new))]
  if (length(miss_cols) > 0) {
    x_new[, miss_cols] <- 0L
    for (mc in miss_cols) {
      # Par défaut, si la variable associée est NA => on code un type de manque
      base_var <- sub("_missing_code$", "", mc)
      # Certaines variables de contexte : plutôt MCAR (oubli / donnée non notée)
      if (grepl("Age_du_cheval|Season|Tiques_semaines|Freq_acces_exterieur|Type_de_cheval|Classe_de_risque|Exterieur",
                mc, ignore.case = TRUE)) {
        if (base_var %in% names(x_new) && is.na(x_new[[base_var]])) x_new[[mc]] <- sample(c(1L, 2L), 1, prob = c(0.7, 0.3))
      }
      # Analyses : plutôt MNAR (non réalisées car pas d'indication)
      if (grepl("ELISA|WB|PCR|NFS|Bilan|Parametres|SAA|Fibrinog|piroplasmose|ehrlichiose|IHC|FISH|Coloration|LCR|CVID|Hypoglob",
                mc, ignore.case = TRUE)) {
        if (base_var %in% names(x_new) && is.na(x_new[[base_var]])) x_new[[mc]] <- sample(c(1L, 2L), 1, prob = c(0.2, 0.8))
      }
    }
  }
  
  # Assurer les types : caractères pour catégorielles
  # FIX : on ne force plus en character ici, on laissera .coerce_like_train reconstruire les factors
  # x_new <- x_new %>% mutate(across(where(is.factor), as.character))
  
  x_new
}

# -----------------------------
# 7.B) Fonction : prédire P(Lyme) pour N nouveaux chevaux
# -----------------------------
predict_new_horses_catboost <- function(n_new = 10, seed = 123, verbose = TRUE) {
  set.seed(seed)
  
  new_list <- vector("list", n_new)
  for (i in seq_len(n_new)) {
    new_list[[i]] <- .generate_one_realistic_horse(df, feature_cols, seed = sample.int(1e9, 1))
  }
  new_df <- bind_rows(new_list)
  
  # FIX : aligner types (numeric double) + facteurs (niveaux train) pour éviter REAL()/integer
  new_df <- .coerce_like_train(new_df, df, feature_cols, cat_cols)
  
  # FIX : utiliser le helper pool (évite warning cat_features meaningless)
  pool_new <- .make_pool(new_df, feature_cols, label_col = NULL, cat_features_idx = cat_features_idx)
  
  p_lyme <- catboost.predict(cat_model_final, pool_new, prediction_type = "Probability")
  
  out <- new_df %>%
    mutate(
      P_Lyme = as.numeric(p_lyme),
      Cat = case_when(
        P_Lyme < 0.10 ~ "Pas Lyme",
        P_Lyme < 0.50 ~ "Lyme possible",
        P_Lyme < 0.90 ~ "Lyme probable",
        TRUE          ~ "Lyme sûr"
      )
    )
  
  if (verbose) {
    cat("\n====================\n")
    cat("Nouveaux chevaux (aperçu) + P(Lyme)\n")
    cat("====================\n")
    
    # FIX erreur "arguments inutilisés" :
    # - dplyr::select doit être explicitement appelé (sinon select() d'un autre package peut être utilisé)
    # - dplyr::all_of doit être explicitement appelé aussi
    print(
      dplyr::select(
        out,
        P_Lyme,
        Cat,
        dplyr::all_of(intersect(
          c("Age_du_cheval","Sexe","Type_de_cheval","Season","Classe de risque",
            "Freq_acces_exterieur_sem","Exterieur_vegetalisé","Tiques_semaines_précédentes",
            "ELISA_pos","WB_pos","PCR_sang_pos",
            "Meningite","Troubles_de_la_demarche","Uveite_bilaterale",
            "Synovite_avec_epanchement_articulaire","Pseudolyphome_cutane"),
          names(out)
        ))
      ) %>%
        head(10)
    )
  }
  
  out
}

# -----------------------------
# 7.C) Exemple : lancer des prédictions sur nouveaux chevaux réalistes
# -----------------------------
pred_new <- predict_new_horses_catboost(n_new = 25, seed = 20260108, verbose = TRUE)

# Optionnel : export des nouveaux chevaux + prédictions
out_new_xlsx <- "C:/Users/q.lamboley/Downloads/CatBoost_new_horses_predictions.xlsx"
openxlsx::write.xlsx(pred_new, out_new_xlsx, overwrite = TRUE)
cat("\nExport nouveaux chevaux :", out_new_xlsx, "\n")


print(cbind(as.numeric(pred_new$P_Lyme),pred_new$Cat))


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


# ============================================================
# 7) Prédire P(Lyme) pour UN cheval (VANILLE DES BALEINES)
#    - ignore colonnes inutiles (Ville/CP/Lat/Long)
#    - NA => MNAR (2) si analyses, sinon MCAR (1)
#    - Cat selon seuils: <0.25 / <0.50 / <0.75 / >=0.75
# ============================================================

# --- Liste des colonnes "analyses" (MNAR si NA)
analysis_cols <- c(
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
)

# --- FIX : aligner les types du "new_df" sur ceux du train (factors + numeric double)
.coerce_like_train <- function(new_df, df_train, feature_cols, cat_cols) {
  num_cols <- setdiff(feature_cols, cat_cols)
  
  # Catégorielles: factor avec niveaux identiques
  for (cc in cat_cols) {
    if (cc %in% names(new_df)) {
      lv <- levels(df_train[[cc]])
      new_df[[cc]] <- as.character(new_df[[cc]])
      new_df[[cc]] <- factor(new_df[[cc]], levels = lv)
    }
  }
  
  # Numériques: toujours numeric (double)
  for (cc in num_cols) {
    if (cc %in% names(new_df)) {
      new_df[[cc]] <- as.numeric(new_df[[cc]])
    }
  }
  
  new_df
}

# --- Helper : convertir "oui/non" -> 1/0 UNIQUEMENT si la colonne est numérique dans df
.yn_to_num_if_needed <- function(val, col_train) {
  if (length(val) == 0) return(val)
  if (is.null(val) || all(is.na(val))) return(val)
  
  if (is.numeric(col_train) || is.integer(col_train)) {
    v <- val
    if (is.character(v) || is.factor(v)) {
      vv <- tolower(trimws(as.character(v)))
      vv[vv %in% c("oui","yes","y","true","vrai")] <- "1"
      vv[vv %in% c("non","no","n","false","faux")] <- "0"
      suppressWarnings(vv <- as.numeric(vv))
      return(vv)
    }
  }
  val
}

# -----------------------------
# 7.A) Construire le cheval unique (avec NA ailleurs)
# -----------------------------
horse_raw <- tibble::tibble(
  Nom_du_Cheval = "TAGADA",
  Age_du_cheval = 19,
  Sexe = "F",
  Type_de_cheval = "Poney_de_selle",
  Season = "Printemps",
  Ville = "Altviller",
  CP = "57730",
  Latitude = NA_real_,     
  Longitude = NA_real_,   
  Classe_de_risque = "fort",   
  
  Exterieur_vegetalisé = "oui",
  Freq_acces_exterieur_sem = 3,
  Tiques_semaines_précédentes = "oui",  
  Examen_clinique = "oui",
  
  piroplasmose_neg = "oui",
  ehrlichiose_negatif = "oui",
  Bilan_sanguin_normal = "oui",
  NFS_normale = "oui",
  Parametres_musculaires_normaux = "oui",
  Parametres_renaux_normaux = "oui",
  Parametres_hepatiques_normaux = "oui",
  SAA_normal = "oui",
  Fibrinogène_normal = "oui",
  
  ELISA_pos = "oui",
  ELISA_OspA_pos = "non",
  ELISA_OspF_pos = "non",
  ELISA_p39 = "non",
  WB_pos = "oui",
  PCR_sang_pos = "non",
  
  Meningite = "non",
  Radiculonevrite = "non",
  Troubles_de_la_demarche = "non",
  Dysphagie = "non",
  Fasciculations_musculaires = "non",
  Uveite_bilaterale = "non",
  Cecite_avec_cause_inflammatoire = "non",
  Synechies = "non",
  Atrophie = "non",
  Dyscories = "non",
  Myosis = "non",
  Synovite_avec_epanchement_articulaire = "non",
  Pseudolyphome_cutane = "oui",
  Pododermatite = "oui"
)


# --- Nettoyage colonnes inutiles (comme demandé)
horse_raw <- horse_raw %>%
  dplyr::select(-any_of(c("Ville","CP","Latitude","Longitude")))

# --- Renommages vers la convention du modèle
rename_map <- c(
  "Classe_de_risque" = "Classe de risque",
  "ehrlichiose_negatif" = "ehrlichiose_neg"
)
for (nm in names(rename_map)) {
  if (nm %in% names(horse_raw) && !(rename_map[[nm]] %in% names(horse_raw))) {
    names(horse_raw)[names(horse_raw) == nm] <- rename_map[[nm]]
  }
}

# --- ID lisible à part
ids <- if ("Nom_du_Cheval" %in% names(horse_raw)) horse_raw$Nom_du_Cheval else "Cheval_1"

# --- On retire Nom_du_Cheval de la matrice de features (comme l'entraînement)
horse_src <- horse_raw %>% dplyr::select(-any_of("Nom_du_Cheval"))

# --- Template EXACT attendu par CatBoost = toutes les feature_cols
x_pred <- df[0, feature_cols, drop = FALSE]
x_pred <- x_pred[rep(1, 1), , drop = FALSE]
x_pred[,] <- NA

# --- Remplir ce qu'on a (intersection)
common_cols <- intersect(names(horse_src), names(x_pred))
for (cc in common_cols) {
  x_pred[[cc]] <- horse_src[[cc]]
}

# --- Important : tu veux NA dans "Classe de risque"
if ("Classe de risque" %in% names(x_pred)) {
  x_pred[["Classe de risque"]] <- NA
}

# --- Conversion oui/non -> 1/0 uniquement si la colonne est numérique dans df
for (cc in intersect(names(x_pred), names(df))) {
  x_pred[[cc]] <- .yn_to_num_if_needed(x_pred[[cc]], df[[cc]])
}

# -----------------------------
# 7.B) Remplir automatiquement les *_missing_code selon les NA
#      - NA sur analyse => 2 (MNAR)
#      - NA hors analyse => 1 (MCAR)
#      - non NA => 0
# -----------------------------
miss_cols <- names(x_pred)[grepl("_missing_code$", names(x_pred))]
if (length(miss_cols) > 0) {
  
  # init à 0
  x_pred[, miss_cols] <- 0L
  
  for (mc in miss_cols) {
    base <- sub("_missing_code$", "", mc)
    
    if (base %in% names(x_pred)) {
      is_miss <- is.na(x_pred[[base]])
      
      if (any(is_miss)) {
        code_val <- if (base %in% analysis_cols) 2L else 1L
        x_pred[[mc]][is_miss] <- code_val
      }
    }
  }
}

# -----------------------------
# 7.C) Aligner types EXACTEMENT comme df (factors + numeric double)
# -----------------------------
x_pred <- .coerce_like_train(x_pred, df, feature_cols, cat_cols)

# -----------------------------
# 7.D) Prédiction CatBoost + seuils
# -----------------------------
pool_one <- .make_pool(x_pred, feature_cols, label_col = NULL, cat_features_idx = cat_features_idx)
p_one <- catboost.predict(cat_model_final, pool_one, prediction_type = "Probability")

res_one <- tibble::tibble(
  ID = ids,
  P_Lyme = as.numeric(p_one),
  Cat = dplyr::case_when(
    as.numeric(p_one) < 0.25 ~ "Pas de Lyme ou informations insuffisantes",
    as.numeric(p_one) < 0.50 ~ "Lyme possible",
    as.numeric(p_one) < 0.75 ~ "Lyme probable",
    TRUE                     ~ "Lyme sûr"
  )
)

print(res_one)

print(cbind(as.numeric(res_one$P_Lyme), res_one$Cat))

message(str_c("TAGADA se situe dans la catégorie",res_one$Cat, sep = " "))







catboost.save_model(
  cat_model_final,
  "C:/Users/q.lamboley/Downloads/equine_lyme_catboost.cbm"
)

if (!requireNamespace("jsonlite", quietly=TRUE)) install.packages("jsonlite")

meta <- list(
  feature_cols = feature_cols,
  cat_cols = cat_cols,
  factor_levels = lapply(cat_cols, function(cc) levels(df[[cc]]))
)
names(meta$factor_levels) <- cat_cols

jsonlite::write_json(
  meta,
  "C:/Users/q.lamboley/Downloads/equine_lyme_catboost_meta.json",
  pretty = TRUE, auto_unbox = TRUE
)




































































































































































































































































































# # -----------------------------
# # 1) Récupérer / générer les données
# # -----------------------------
# if (exists("df_demo")) {
#   df <- df_demo
# } else if (exists("generate_equine_lyme_perfect_dataset")) {
#   df <- generate_equine_lyme_perfect_dataset(
#     n_per_class = 1000,
#     random_state = 123,
#     add_missingness = TRUE,
#     keep_internal_label = TRUE
#   )
# } else {
#   stop("Je ne trouve ni 'df_demo' ni la fonction 'generate_equine_lyme_perfect_dataset()'. Exécute d'abord ton script générateur.")
# }
# 
# if (!("Lyme_true" %in% names(df))) stop("La colonne cible 'Lyme_true' est absente. Génère avec keep_internal_label = TRUE.")
# 
# # -----------------------------
# # 2) Pré-traitements (CatBoost-friendly)
# # -----------------------------
# # - On enlève Nom_du_Cheval (risque de fuite d'info).
# # - On crée une cible binaire 0/1 (catboost aime bien).
# # - On laisse les NA tels quels (CatBoost les gère).
# # - IMPORTANT CatBoost R :
# #   * les catégorielles doivent être en factor (sinon cat_features ignoré)
# #   * les numériques doivent être en numeric (double), pas integer (évite REAL()/integer)
# # - On ajoute un row_id stable pour recoller les OOF sans bidouille de rownames().
# df <- df %>%
#   dplyr::select(-any_of("Nom_du_Cheval")) %>%
#   mutate(
#     Lyme_y = as.integer(Lyme_true == 1L),
#     row_id = dplyr::row_number()
#   )
# 
# # Liste des features = tout sauf la cible (et surtout PAS row_id, sinon fuite d'info)
# feature_cols <- setdiff(names(df), c("Lyme_true", "Lyme_y", "row_id"))
# 
# # --- FIX demandé : empêcher "Sexe" d'être un facteur dominant -> on le retire des features (comme Nom_du_Cheval)
# feature_cols <- setdiff(feature_cols, "Sexe")
# 
# # Détecter les colonnes catégorielles (character OU factor) puis les forcer en factor
# cat_cols <- feature_cols[sapply(df[, feature_cols, drop = FALSE], function(x) is.character(x) || is.factor(x))]
# df <- df %>%
#   mutate(across(all_of(cat_cols), ~ as.factor(.x)))
# 
# # Forcer toutes les colonnes non-catégorielles en numeric (double) (évite REAL()/integer)
# num_cols <- setdiff(feature_cols, cat_cols)
# df <- df %>%
#   mutate(across(all_of(num_cols), ~ as.numeric(.x)))
# 
# # Indices (0-based) des variables catégorielles pour CatBoost
# cat_features_idx <- which(feature_cols %in% cat_cols) - 1L  # CatBoost R = indices 0-based
# 
# # -----------------------------
# # 3) Helper : construire un pool CatBoost
# # -----------------------------
# .make_pool <- function(df_sub, feature_cols, label_col = "Lyme_y", cat_features_idx = integer(0)) {
#   X <- df_sub[, feature_cols, drop = FALSE]
#   y <- if (!is.null(label_col) && label_col %in% names(df_sub)) df_sub[[label_col]] else NULL
#   
#   # data.frame => CatBoost utilise les types (factor) et ignore cat_features
#   if (is.data.frame(X)) {
#     if (is.null(y)) catboost.load_pool(data = X) else catboost.load_pool(data = X, label = y)
#   } else {
#     if (is.null(y)) catboost.load_pool(data = X, cat_features = cat_features_idx)
#     else            catboost.load_pool(data = X, label = y, cat_features = cat_features_idx)
#   }
# }
# 
# # -----------------------------
# # 4) Validation croisée 5 folds (stratifiée)
# # -----------------------------
# set.seed(20260108)
# folds <- vfold_cv(df, v = 5, strata = Lyme_y)
# 
# # Paramètres CatBoost (robustes par défaut)
# # - loss_function = Logloss pour proba
# # - eval_metric = AUC
# # - iterations élevé + early stopping pour ne pas overfit
# # - depth 6-8 souvent bon
# # - l2_leaf_reg régularisation
# params <- list(
#   loss_function = "Logloss",
#   eval_metric = "AUC",
#   iterations = 4000,
#   learning_rate = 0.05,
#   depth = 8,
#   l2_leaf_reg = 6,
#   random_seed = 20260108,
#   # overfitting detector
#   od_type = "Iter",
#   od_wait = 150,
#   # pas de logs trop verbeux
#   logging_level = "Silent",
#   # meilleure proba
#   # (CatBoost est souvent déjà bien calibré, mais ça aide parfois)
#   # leaf_estimation_iterations = 10
#   allow_writing_files = FALSE
# )
# 
# fold_metrics <- vector("list", length = 5)
# names(fold_metrics) <- paste0("Fold_", seq_len(5))
# 
# # On conserve aussi les probabilités out-of-fold pour tracer/choisir seuils si besoin
# oof_pred <- rep(NA_real_, nrow(df))
# 
# for (i in seq_len(5)) {
#   cat("\n====================\n")
#   cat("Fold", i, "/ 5\n")
#   cat("====================\n")
#   
#   split_i <- folds$splits[[i]]
#   train_raw <- analysis(split_i)
#   test_raw  <- assessment(split_i)
#   
#   pool_train <- .make_pool(train_raw, feature_cols, "Lyme_y", cat_features_idx)
#   pool_test  <- .make_pool(test_raw,  feature_cols, "Lyme_y", cat_features_idx)
#   
#   model_i <- catboost.train(
#     learn_pool = pool_train,
#     test_pool  = pool_test,
#     params = params
#   )
#   
#   # proba classe 1 (= Lyme)
#   prob_lyme <- catboost.predict(model_i, pool_test, prediction_type = "Probability")
#   
#   # stocker OOF proba (utile pour calibration + seuils) — CORRIGÉ via row_id
#   oof_pred[test_raw$row_id] <- prob_lyme
#   
#   pred_class <- ifelse(prob_lyme >= 0.5, 1L, 0L)
#   y_true <- test_raw$Lyme_y
#   
#   TP <- sum(pred_class == 1L & y_true == 1L)
#   TN <- sum(pred_class == 0L & y_true == 0L)
#   FP <- sum(pred_class == 1L & y_true == 0L)
#   FN <- sum(pred_class == 0L & y_true == 1L)
#   
#   acc  <- (TP + TN) / (TP + TN + FP + FN)
#   sens <- if ((TP + FN) == 0) NA else TP / (TP + FN)
#   spec <- if ((TN + FP) == 0) NA else TN / (TN + FP)
#   
#   roc_obj <- tryCatch(
#     pROC::roc(response = y_true, predictor = prob_lyme, levels = c(0, 1), quiet = TRUE),
#     error = function(e) NULL
#   )
#   auc <- if (is.null(roc_obj)) NA else as.numeric(pROC::auc(roc_obj))
#   
#   fold_metrics[[i]] <- tibble(
#     fold = i,
#     n_train = nrow(train_raw),
#     n_test = nrow(test_raw),
#     accuracy = acc,
#     sensitivity = sens,
#     specificity = spec,
#     auc = auc,
#     TP = TP, TN = TN, FP = FP, FN = FN
#   )
#   
#   print(fold_metrics[[i]])
# }
# 
# metrics_df <- bind_rows(fold_metrics)
# 
# cat("\n====================\n")
# cat("Résumé CV (5 folds)\n")
# cat("====================\n")
# print(metrics_df)
# 
# summary_df <- metrics_df %>%
#   summarise(
#     accuracy_mean = mean(accuracy, na.rm = TRUE),
#     accuracy_sd   = sd(accuracy, na.rm = TRUE),
#     sensitivity_mean = mean(sensitivity, na.rm = TRUE),
#     sensitivity_sd   = sd(sensitivity, na.rm = TRUE),
#     specificity_mean = mean(specificity, na.rm = TRUE),
#     specificity_sd   = sd(specificity, na.rm = TRUE),
#     auc_mean = mean(auc, na.rm = TRUE),
#     auc_sd   = sd(auc, na.rm = TRUE)
#   )
# 
# cat("\n====================\n")
# cat("Moyenne ± SD\n")
# cat("====================\n")
# print(summary_df)
# 
# # -----------------------------
# # 4bis) Fix robuste des rownames (important pour OOF + prédictions)
# # -----------------------------
# # rsample conserve les rownames ; on les force pour être sûr
# # (utile si tu veux recalculer des seuils sur oof_pred après coup)
# if (any(is.na(oof_pred))) {
#   cat("\n[INFO] Certaines proba OOF sont NA -> on reconstruit en forçant les rownames.\n")
#   df <- df %>% mutate(.row_id = seq_len(n()))
#   rownames(df) <- as.character(df$.row_id)
#   df$.row_id <- NULL
#   # On peut relancer la CV si tu veux OOF complet (optionnel).
# }
# 
# # -----------------------------
# # 5) Entraîner un modèle final sur tout le dataset + importance variables
# # -----------------------------
# pool_all <- .make_pool(df, feature_cols, "Lyme_y", cat_features_idx)
# 
# cat_model_final <- catboost.train(
#   learn_pool = pool_all,
#   params = params
# )
# 
# # Importance : PredictionValuesChange est souvent une bonne option.
# # (plus robuste que "gain" selon contextes)
# imp <- catboost.get_feature_importance(
#   cat_model_final,
#   pool = pool_all,
#   type = "PredictionValuesChange"
# )
# 
# varimp_df <- tibble(
#   variable = feature_cols,
#   importance = as.numeric(imp)
# ) %>%
#   arrange(desc(importance))
# 
# cat("\n====================\n")
# cat("Top 20 importances (modèle final CatBoost)\n")
# cat("====================\n")
# print(head(varimp_df, 20))
# 
# # -----------------------------
# # 6) Export (optionnel)
# # -----------------------------
# out_xlsx <- "C:/Users/q.lamboley/Downloads/CatBoost_CV5_equine_lyme_results.xlsx"
# wb <- createWorkbook()
# 
# addWorksheet(wb, "CV_folds")
# writeData(wb, "CV_folds", metrics_df)
# 
# addWorksheet(wb, "CV_summary")
# writeData(wb, "CV_summary", summary_df)
# 
# addWorksheet(wb, "VarImp_final")
# writeData(wb, "VarImp_final", varimp_df)
# 
# saveWorkbook(wb, out_xlsx, overwrite = TRUE)
# cat("\nRésultats exportés :", out_xlsx, "\n")
# 
# # ============================================================
# # 7) Prédire P(Lyme) pour UN cheval (VANILLE DES BALEINES)
# #    - ignore colonnes inutiles (Ville/CP/Lat/Long)
# #    - NA => MNAR (2) si analyses, sinon MCAR (1)
# #    - Cat selon seuils: <0.10 / <0.50 / <0.90 / >=0.90
# # ============================================================
# 
# # --- Liste des colonnes "analyses" (MNAR si NA)
# analysis_cols <- c(
#   # bilans / exclusions
#   "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
#   "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
#   "SAA_normal","Fibrinogène_normal",
#   
#   # sérologies / PCR sang
#   "ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","WB_pos","PCR_sang_pos","SNAP_C6_pos","IFAT_pos",
#   
#   # examens ciblés / PCR locales / LCR
#   "PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos",
#   "PCR_liquide_articulaire_pos","LCR_pleiocytose","LCR_proteines_augmentees",
#   
#   # histo / marquages
#   "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos",
#   
#   # immuno
#   "CVID","Hypoglobulinemie"
# )
# 
# # --- FIX : aligner les types du "new_df" sur ceux du train (factors + numeric double)
# .coerce_like_train <- function(new_df, df_train, feature_cols, cat_cols) {
#   num_cols <- setdiff(feature_cols, cat_cols)
#   
#   # Catégorielles: factor avec niveaux identiques
#   for (cc in cat_cols) {
#     if (cc %in% names(new_df)) {
#       lv <- levels(df_train[[cc]])
#       new_df[[cc]] <- as.character(new_df[[cc]])
#       new_df[[cc]] <- factor(new_df[[cc]], levels = lv)
#     }
#   }
#   
#   # Numériques: toujours numeric (double)
#   for (cc in num_cols) {
#     if (cc %in% names(new_df)) {
#       new_df[[cc]] <- as.numeric(new_df[[cc]])
#     }
#   }
#   
#   new_df
# }
# 
# # --- Helper : convertir "oui/non" -> 1/0 UNIQUEMENT si la colonne est numérique dans df
# .yn_to_num_if_needed <- function(val, col_train) {
#   if (length(val) == 0) return(val)
#   if (is.null(val) || all(is.na(val))) return(val)
#   
#   if (is.numeric(col_train) || is.integer(col_train)) {
#     v <- val
#     if (is.character(v) || is.factor(v)) {
#       vv <- tolower(trimws(as.character(v)))
#       vv[vv %in% c("oui","yes","y","true","vrai")] <- "1"
#       vv[vv %in% c("non","no","n","false","faux")] <- "0"
#       suppressWarnings(vv <- as.numeric(vv))
#       return(vv)
#     }
#   }
#   val
# }
# 
# # -----------------------------
# # 7.A) Construire le cheval unique (avec NA ailleurs)
# # -----------------------------
# horse_raw <- tibble::tibble(
#   Nom_du_Cheval = "SHIRA",
#   Age_du_cheval = 19,
#   Sexe = "F",
#   Type_de_cheval = "Poney_de_selle",
#   Season = "Printemps",
#   Ville = "Altviller",
#   CP = "57730",
#   Latitude = NA_real_,     # tu avais "MCAR" -> ici on met NA (de toute façon supprimé)
#   Longitude = NA_real_,    # idem
#   Classe_de_risque = NA,   # demandé : NA
#   
#   Exterieur_vegetalisé = "oui",
#   Freq_acces_exterieur_sem = 7,
#   Tiques_semaines_précédentes = "oui",  # si tu veux NA -> mets NA (et ce sera MCAR automatiquement)
#   Examen_clinique = "oui",
#   
#   piroplasmose_neg = "oui",
#   ehrlichiose_negatif = "oui",
#   Bilan_sanguin_normal = "oui",
#   NFS_normale = "oui",
#   Parametres_musculaires_normaux = "oui",
#   Parametres_renaux_normaux = "oui",
#   Parametres_hepatiques_normaux = "oui",
#   SAA_normal = "oui",
#   Fibrinogène_normal = "oui",
#   
#   ELISA_pos = "non",
#   ELISA_OspA_pos = "oui",
#   ELISA_OspF_pos = "non",
#   ELISA_p39 = "oui",
#   WB_pos = "non",
#   PCR_sang_pos = "non",
#   
#   Meningite = "non",
#   Radiculonevrite = "oui",
#   Troubles_de_la_demarche = "non",
#   Dysphagie = "non",
#   Fasciculations_musculaires = "non",
#   Uveite_bilaterale = "non",
#   Cecite_avec_cause_inflammatoire = "non",
#   Synechies = "non",
#   Atrophie = "non",
#   Dyscories = "non",
#   Myosis = "non",
#   Synovite_avec_epanchement_articulaire = "non",
#   Pseudolyphome_cutane = "non",
#   Pododermatite = "non"
# )
# 
# 
# # --- Nettoyage colonnes inutiles (comme demandé)
# horse_raw <- horse_raw %>%
#   dplyr::select(-any_of(c("Ville","CP","Latitude","Longitude")))
# 
# # --- Renommages vers la convention du modèle
# rename_map <- c(
#   "Classe_de_risque" = "Classe de risque",
#   "ehrlichiose_negatif" = "ehrlichiose_neg"
# )
# for (nm in names(rename_map)) {
#   if (nm %in% names(horse_raw) && !(rename_map[[nm]] %in% names(horse_raw))) {
#     names(horse_raw)[names(horse_raw) == nm] <- rename_map[[nm]]
#   }
# }
# 
# # --- ID lisible à part
# ids <- if ("Nom_du_Cheval" %in% names(horse_raw)) horse_raw$Nom_du_Cheval else "Cheval_1"
# 
# # --- On retire Nom_du_Cheval de la matrice de features (comme l'entraînement)
# horse_src <- horse_raw %>% dplyr::select(-any_of("Nom_du_Cheval"))
# 
# # --- Template EXACT attendu par CatBoost = toutes les feature_cols
# x_pred <- df[0, feature_cols, drop = FALSE]
# x_pred <- x_pred[rep(1, 1), , drop = FALSE]
# x_pred[,] <- NA
# 
# # --- Remplir ce qu'on a (intersection)
# common_cols <- intersect(names(horse_src), names(x_pred))
# for (cc in common_cols) {
#   x_pred[[cc]] <- horse_src[[cc]]
# }
# 
# # --- Important : tu veux NA dans "Classe de risque"
# if ("Classe de risque" %in% names(x_pred)) {
#   x_pred[["Classe de risque"]] <- NA
# }
# 
# # --- Conversion oui/non -> 1/0 uniquement si la colonne est numérique dans df
# for (cc in intersect(names(x_pred), names(df))) {
#   x_pred[[cc]] <- .yn_to_num_if_needed(x_pred[[cc]], df[[cc]])
# }
# 
# # -----------------------------
# # 7.B) Remplir automatiquement les *_missing_code selon les NA
# #      - NA sur analyse => 2 (MNAR)
# #      - NA hors analyse => 1 (MCAR)
# #      - non NA => 0
# # -----------------------------
# miss_cols <- names(x_pred)[grepl("_missing_code$", names(x_pred))]
# if (length(miss_cols) > 0) {
#   
#   # init à 0
#   x_pred[, miss_cols] <- 0L
#   
#   for (mc in miss_cols) {
#     base <- sub("_missing_code$", "", mc)
#     
#     if (base %in% names(x_pred)) {
#       is_miss <- is.na(x_pred[[base]])
#       
#       if (any(is_miss)) {
#         code_val <- if (base %in% analysis_cols) 2L else 1L
#         x_pred[[mc]][is_miss] <- code_val
#       }
#     }
#   }
# }
# 
# # -----------------------------
# # 7.C) Aligner types EXACTEMENT comme df (factors + numeric double)
# # -----------------------------
# x_pred <- .coerce_like_train(x_pred, df, feature_cols, cat_cols)
# 
# # -----------------------------
# # 7.D) Prédiction CatBoost + seuils
# # -----------------------------
# pool_one <- .make_pool(x_pred, feature_cols, label_col = NULL, cat_features_idx = cat_features_idx)
# p_one <- catboost.predict(cat_model_final, pool_one, prediction_type = "Probability")
# 
# res_one <- tibble::tibble(
#   ID = ids,
#   P_Lyme = as.numeric(p_one),
#   Cat = dplyr::case_when(
#     as.numeric(p_one) < 0.10 ~ "Pas Lyme",
#     as.numeric(p_one) < 0.50 ~ "Lyme possible",
#     as.numeric(p_one) < 0.90 ~ "Lyme probable",
#     TRUE                     ~ "Lyme sûr"
#   )
# )
# 
# print(res_one)
# print(cbind(as.numeric(res_one$P_Lyme), res_one$Cat))
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # ============================================================
# # 7) Prédire P(Lyme) sur un TABLEAU de chevaux (Excel)
# #    - supprime Ville/CP
# #    - force Classe de risque = NA (tous NA)
# #    - missing_code auto:
# #        * NA hors-analyses => MCAR (1)
# #        * NA analyses      => MNAR (2)
# #    - ajoute les colonnes manquantes attendues par CatBoost
# #    - prédiction CatBoost + catégories seuils fixes (0.10/0.50/0.90)
# # ============================================================
# 
# stopifnot(exists("cat_model_final"), exists("df"), exists("feature_cols"), exists("cat_cols"), exists("cat_features_idx"))
# stopifnot(exists(".make_pool"), exists(".coerce_like_train"))
# 
# suppressWarnings({
#   pkgs <- c("dplyr", "tibble", "openxlsx")
#   for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
# })
# library(dplyr)
# library(tibble)
# library(openxlsx)
# 
# # -----------------------------
# # 7.0) Chemin vers TON fichier (à adapter)
# # -----------------------------
# in_xlsx <- "E:/PhD/Diagnostic_tool/Tableau_results_enquete_prospective.xlsx"
# # (mets ici le chemin exact chez toi)
# 
# df_src <- openxlsx::read.xlsx(in_xlsx)
# 
# # -----------------------------
# # 7.1) Nettoyage colonnes inutiles + renommages nécessaires
# # -----------------------------
# # On enlève Ville / CP (et aussi Latitude/Longitude comme avant)
# df_src <- df_src %>%
#   dplyr::select(-any_of(c("Ville", "CP", "Latitude", "Longitude")))
# 
# # On renomme vers les noms du générateur / modèle
# # (dans ton tableau, "Classe_de_risque" au lieu de "Classe de risque"
# #  et "ehrlichiose_negatif" au lieu de "ehrlichiose_neg")
# rename_map <- c(
#   "Classe_de_risque" = "Classe de risque",
#   "ehrlichiose_negatif" = "ehrlichiose_neg"
# )
# for (nm in names(rename_map)) {
#   if (nm %in% names(df_src) && !(rename_map[[nm]] %in% names(df_src))) {
#     names(df_src)[names(df_src) == nm] <- rename_map[[nm]]
#   }
# }
# 
# # Force Classe de risque = NA pour tout le monde (même si déjà renseigné)
# if ("Classe de risque" %in% names(df_src)) {
#   df_src[["Classe de risque"]] <- NA
# }
# 
# # -----------------------------
# # 7.2) Construire un dataset "prédiction" exactement compatible CatBoost
# #      -> mêmes features que le modèle (feature_cols)
# # -----------------------------
# # On garde un identifiant lisible à côté (si présent)
# id_col <- if ("Nom_du_Cheval" %in% names(df_src)) "Nom_du_Cheval" else NULL
# ids <- if (!is.null(id_col)) df_src[[id_col]] else paste0("Cheval_", seq_len(nrow(df_src)))
# 
# # Template : 0 lignes, colonnes = features du modèle
# x_template <- df[0, feature_cols, drop = FALSE]
# 
# # Initialiser x_pred (N lignes, toutes colonnes du modèle)
# x_pred <- x_template[rep(1, nrow(df_src)), , drop = FALSE]
# x_pred[,] <- NA
# 
# # Remplir x_pred avec les colonnes disponibles du tableau (intersection)
# common_cols <- intersect(names(df_src), names(x_pred))
# for (cc in common_cols) {
#   x_pred[[cc]] <- df_src[[cc]]
# }
# 
# # Re-force Classe de risque = NA côté x_pred aussi (sécurité)
# if ("Classe de risque" %in% names(x_pred)) {
#   x_pred[["Classe de risque"]] <- NA
# }
# 
# # -----------------------------
# # 7.3) Définir quelles colonnes sont des "analyses" (MNAR si NA)
# #      (reprend l’esprit de ton générateur)
# # -----------------------------
# analysis_cols <- c(
#   # bilans / exclusions
#   "piroplasmose_neg","ehrlichiose_neg","Bilan_sanguin_normal","NFS_normale",
#   "Parametres_musculaires_normaux","Parametres_renaux_normaux","Parametres_hepatiques_normaux",
#   "SAA_normal","Fibrinogène_normal",
# 
#   # sérologies / PCR sang
#   "ELISA_pos","ELISA_OspA_pos","ELISA_OspF_pos","ELISA_p39","WB_pos","PCR_sang_pos","SNAP_C6_pos","IFAT_pos",
# 
#   # examens ciblés / PCR locales / LCR
#   "PCR_LCR_pos","PCR_synoviale_pos","PCR_peau_pos","PCR_humeur_aqueuse_pos","PCR_tissu_nerveux_pos",
#   "PCR_liquide_articulaire_pos","LCR_pleiocytose","LCR_proteines_augmentees",
# 
#   # histo / marquages
#   "IHC_tissulaire_pos","Coloration_argent_pos","FISH_tissulaire_pos",
# 
#   # immuno
#   "CVID","Hypoglobulinemie"
# )
# 
# # -----------------------------
# # 7.4) Remplir automatiquement les *_missing_code selon les NA
# #      - NA sur analyse => 2 (MNAR)
# #      - NA hors analyse => 1 (MCAR)
# #      - non NA => 0
# # -----------------------------
# miss_cols <- names(x_pred)[grepl("_missing_code$", names(x_pred))]
# if (length(miss_cols) > 0) {
# 
#   # init à 0
#   x_pred[, miss_cols] <- 0L
# 
#   for (mc in miss_cols) {
#     base <- sub("_missing_code$", "", mc)
# 
#     if (base %in% names(x_pred)) {
#       is_miss <- is.na(x_pred[[base]])
# 
#       if (any(is_miss)) {
#         code_val <- if (base %in% analysis_cols) 2L else 1L
#         x_pred[[mc]][is_miss] <- code_val
#       }
#     }
#   }
# }
# 
# # -----------------------------
# # 7.5) Aligner les types comme le train (CatBoost)
# #      - facteurs (niveaux train)
# #      - numériques en double
# # -----------------------------
# # FIX : on aligne strictement sur df (train) via le helper déjà utilisé pour les nouveaux chevaux
# x_pred <- .coerce_like_train(x_pred, df, feature_cols, cat_cols)
# 
# # -----------------------------
# # 7.6) Prédiction P(Lyme) CatBoost + catégories seuils fixes
# # -----------------------------
# # FIX : utiliser le helper pool (évite warning cat_features meaningless)
# pool_pred <- .make_pool(x_pred, feature_cols, label_col = NULL, cat_features_idx = cat_features_idx)
# 
# p_lyme <- catboost.predict(cat_model_final, pool_pred, prediction_type = "Probability")
# 
# results_pred <- tibble::tibble(
#   ID = ids,
#   P_Lyme = as.numeric(p_lyme),
#   Cat = dplyr::case_when(
#     P_Lyme < 0.10 ~ "Pas Lyme",
#     P_Lyme < 0.50 ~ "Lyme possible",
#     P_Lyme < 0.90 ~ "Lyme probable",
#     TRUE          ~ "Lyme sûr"
#   )
# )
# 
# print(head(results_pred, 20))
# 
# # -----------------------------
# # 7.7) Export Excel (optionnel)
# # -----------------------------
# out_xlsx <- "C:/Users/q.lamboley/Downloads/Predictions_Lyme_CatBoost_sur_tableau.xlsx"
# wb <- createWorkbook()
# 
# addWorksheet(wb, "Input_table_nettoye")
# writeData(wb, "Input_table_nettoye", df_src)
# 
# addWorksheet(wb, "X_pred_modele")
# writeData(wb, "X_pred_modele", x_pred)
# 
# addWorksheet(wb, "Predictions")
# writeData(wb, "Predictions", results_pred)
# 
# saveWorkbook(wb, out_xlsx, overwrite = TRUE)
# cat("\nExport OK :", out_xlsx, "\n")
# 
# 
# print(cbind(as.numeric(results_pred$P_Lyme), results_pred$Cat))
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
