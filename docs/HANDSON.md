# CorSeaCare_yolo — Mémoire hands-on (méthodo · installation · modèle · pratique)

Document de référence opérationnel pour reprendre le projet sur **n'importe quelle machine**.
Couvre l'ensemble de la démarche, avec un focus détaillé (§9–§11) sur le travail récent :
publication du dépôt, Git LFS, et couche Label Studio reproductible.

- **Dépôt :** https://github.com/lfnothias/microplastic_detection_csc26
- **Licences :** code **AGPL-3.0**, données (images + annotations) **CC0**
- **Plateforme :** macOS Apple Silicon (MPS) ; `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **Mission :** CorSeaCare / *Mare Vivu* (Corse) — sciences participatives microplastiques

---

## 0. Vue d'ensemble

Détecter, **compter**, **classer** (5 morphotypes plastiques + `autre`) et **mesurer** (taille mm,
couleur) les particules de microplastique sur des **photos de tamis** de filet Manta. Outil
**local**, autour d'une boucle **human-in-the-loop** (annotation assistée → entraînement →
ré-annotation active).

**6 classes :** `fragment` · `fibre` · `film` · `mousse` · `pellet` · `autre` (organique/indéterminé).

**Statut :** preuve de concept recherche. Le pipeline complet fonctionne de bout en bout ;
la *précision* est limitée par le faible nombre de tamis distincts annotés (le levier principal).

**Résolution :** 12 MP suffisent (pas besoin du RAW 48 MP) — les particules font ~1–2 % du cadre.

---

## 1. Installation (clone reproductible)

**Prérequis :** `uv` (`brew install uv`), `git`, `git-lfs` (`brew install git-lfs`).

```bash
git lfs install
git clone https://github.com/lfnothias/microplastic_detection_csc26.git
cd microplastic_detection_csc26 && git lfs pull   # récupère images + poids (LFS)
./scripts/setup.sh                                # uv sync --extra dev + uv tool install label-studio
```

Pour **entraîner / faire tourner les vrais modèles** (tire PyTorch, ~Go — bonne connexion) :
```bash
uv pip install "ultralytics>=8.3" "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

**Checkpoint SAM2** (segmentation, pour `count`) : `sam2.1_hiera_tiny.pt` (~149 Mo) se télécharge
séparément (placé dans `data/models/`, ignoré par git via `*.pt`). Non redistribué dans le dépôt.

**Essai sans données ni modèle :**
```bash
uv run pytest                                     # 69 tests (engine)
uv run python scripts/make_test_dataset.py
CORSEACARE_FAKE=1 uv run corseacare count data/test_run/images --out counts.csv --mm-per-px 0.1
```
`CORSEACARE_FAKE=1` remplace les détecteurs par des stubs déterministes (aucun modèle requis).

---

## 2. Données — ce qui est partagé / ce qui ne l'est pas

**Partagé sur le dépôt (reproductible) :**
- `data/corseacare/*.jpg,*.JPG` — **59 images** (TAMIS + 7 MANTA) via **Git LFS**
- `annotations/*.txt` — **11 images annotées** (YOLO)
- `samples.csv` — manifeste : `image,sample_id,split,sieve_mm,px_per_mm,date,location,gps,notes`
- `models/*.pt` — **poids v6 + v7** via LFS
- `labelstudio/*.json` — snapshots des projets Label Studio (git normal, texte)

**NON partagé — régénérable** (recréé par les scripts) : `data/ls_export_yolo`, `data/ls_tiles_yolo`,
`data/corseacare_pred*`, `data/active`, `data/enrich`, `data/reannot_B`, `data/clusters`,
`data/test_run*`, `.venv`, sorties CLI racine.

**NON partagé — volontaire (lourd/hors-scope)** : `data/raw_dng/` (417 Mo, RAW), `data/runs/`
(186 Mo, tous les entraînements **sauf** les 2 meilleurs poids publiés), `data/models/sam2*.pt`
(149 Mo), `docs/superpowers/` (specs/plans de travail).

**Hors dépôt par nature** : la base `label_studio.sqlite3` (contient email + hash mdp + token API
+ chemins absolus → jamais committée ; voir §9).

**Le seul vrai « manque »** : les annotations **TAMIS_E/F** en cours de révision dans Label Studio,
pas encore exportées vers `annotations/` + committées.

---

## 3. Méthodologie — de la boîte à la classe (détection)

Pipeline d'inférence tuilée, paramètres « v5 » figés dans `Config` et `scripts/predict_tiled.py` :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `tile_size` | **640** | tuiles SAHI (les particules sont minuscules) |
| `tile_overlap` | **0.5** | recouvrement des tuiles |
| `conf` (inférence) | **0.12** | seuil bas, orienté **rappel** (défaut `Config`=0.25) |
| NMS IoU | **0.5** | fusion des boîtes après recollage |
| `roi_gate` | **True** | garde uniquement l'intérieur du tamis |
| `roi_margin` | **1.0** | jette les boîtes au-delà du bord du tamis |
| `max_box_frac` | **0.04** | jette les boîtes > 4 % de l'image (FP géants / plaques) |

**Étapes (`predict_tiled.py`) :**
1. `tile_origins(W,H,640,0.5)` → découpe en tuiles (padding si bord).
2. `model.predict(tile)` par tuile → `offset_boxes_from_tile` (recolle aux coords image).
3. `nms(dets, iou=0.5)` → fusion.
4. `detect_sieve_circle(img)` → cercle du tamis (Hough multi-passes `p2 ∈ {60,45,32,24}`,
   `minR=0.20·min(W,H)`, `maxR=0.55·min(W,H)`, 1er hit gagne, sinon cercle centré par défaut).
5. `keep_inside_circle(dets, circle, margin=1.0)` → enlève rebord/plateau/outils.
6. Filtre taille : enlève les boîtes dont l'aire > `0.04·H·W`.
7. Overlay (cercle ROI jaune + boîtes colorées par classe) + `counts.csv`.

**Diagnostic « petites particules noires manquées »** : la vérité-terrain ne contient que **4,4 %**
de noir (76/1718) → c'est un **problème de détection** (YOLO n'a jamais appris le noir), pas de
classification. Racine : le proposeur de candidats basé sur la **saturation** ne sait pas proposer
les objets noirs (faible saturation). → TODO §14 : proposeur de candidats sombres.

---

## 4. Calibration mm — le maillage comme règle

Le tamis a un **maillage périodique connu** (1 mm ou 2 mm d'ouverture) servant de règle intégrée.

- `detect_mesh_period` (FFT 2D) → période en px du grillage
- `solve_pitch_mm` / `px_per_mm_from_period` → px↔mm (pitch = ouverture + fil)
- Deux tailles de tamis : **rouge = 1 mm, vert = 2 mm** (tags couleur Finder)

```bash
uv run python scripts/calibrate_mesh.py calibrate   # -> mesh_calibration.json (pitch par maille), 1 fois
uv run python scripts/calibrate_mesh.py apply        # -> remplit px_per_mm dans samples.csv
```
Là où le maillage est trop occulté, l'échelle de cette photo est laissée **vide** (pas devinée).

---

## 5. Mesure & reporting

```bash
uv run corseacare count data/corseacare --out counts.csv --tiled   # détecte+segmente(SAM2)+mesure
uv run corseacare report --particles counts.csv                    # synthèse par échantillon
```
`report` agrège en **par-échantillon** (`report.json` + `report_by_sample.csv`) :
- **comptes par classe** + par couleur
- **taille** : Feret max moyen/médian (mm) + histogramme `<1 / 1–2 / 2–5 / 5–10 / >10` mm
- **aire projetée** (mm², robuste) ; **estimation de volume** (mm³, clairement étiquetée *estimation*)
- **concentration** (particules/m³) si `--tow-volume-m3` fourni

⚠️ Plusieurs photos d'un même tamis = **même matière redistribuée** (on ajoute de l'eau pour
re-répartir les particules) → **non sommées** ; la vue représentative (compte médian) est résumée
(`--sum-views` pour forcer). Taille/couleur nécessitent la segmentation (`count`) ; un CSV
détection-seule ne donne que les comptes. Le **type de polymère n'est PAS inféré** (faut FTIR/Raman).

---

## 6. Modèles

| Fichier (LFS) | Entraîné sur | Tenu à l'écart | Usage |
|---|---|---|---|
| **`models/corseacare_tiles_v7.pt`** | MANTA + TAMIS_DSLR + TAMIS_C (**annotations enrichies**) | TAMIS_B | **recommandé** |
| `models/corseacare_tiles_v6.pt` | mêmes tamis, annotations originales | TAMIS_B | comparaison v6 vs v7 |

- Architecture : **YOLO11n** (Ultralytics, AGPL-3.0), ~5,2 Mo chacun.
- Perf tenue à l'écart : **mAP@50 ≈ 0,37** sur un tamis typique ; le **rappel** est le facteur limitant.
- Géométrie de détection figée = §3. Preuve de concept, pas un instrument calibré.

**Inférence directe (sans ré-entraîner) :**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/predict_tiled.py models/corseacare_tiles_v7.pt 0.12
```

**Ré-entraîner depuis le dépôt :**
```bash
uv run python scripts/build_yolo_dataset.py   # images + annotations + split -> dataset YOLO (sans Label Studio)
uv run python scripts/tile_dataset.py          # découpe en tuiles  (vide d'abord le dossier de sortie !)
uv run python scripts/train_tiles.py           # YOLO11n sur tuiles (MPS, early-stop)
```

---

## 7. Évaluation honnête (anti-overfitting)

**Principe : entraînement et test totalement décorrélés** — aucun tamis physique partagé.
- Split via colonne explicite **`split`** dans `samples.csv` (sample-aware, pas image-aware).
- **val = tamis TAMIS in-domain** (ex. TAMIS_B) ; train = MANTA + TAMIS_DSLR + TAMIS_C (+D).
- « in-domain » = **redistribution spatiale physique** : on remet de l'eau dans le tamis pour
  re-répartir les particules → vues différentes de la *même* matière (cohérence inter-vue).

**Résultats clés :**
- mAP50 inter-tamis : **0,14** (DSLR atypique) → **0,37** (B typique).
- **Plastique vs organique : ~92 %** (étant donné la détection).
- Enrichissement : précision **0,46 → 0,72** à mAP égal.
- **74 % des « faux positifs » sont de vraies particules non annotées** → annotation ~34 %
  incomplète → précision réelle **~86 %**.

**Outils :** `eval_binary.py` (plastique vs organique), `view_consistency.py` (reproductibilité
inter-vue), `results_consistency.py`. Détails : `docs/RESULTS.md`, `docs/RESULTS_v2.md`.

**Piège corrigé :** `tile_dataset.py` ne vidait pas le dossier de sortie → tuiles périmées →
fuite du tamis de val dans le train (le mAP 0,87 de tiles_v4 était **bidon**). Fix : `rmtree` en tête.

---

## 8. Annotation assistée & apprentissage actif

Boucle : les modèles de fondation **proposent**, l'humain **valide** dans Label Studio.

- **Pré-annotation fraîche :** `preannotate_corseacare.py` (candidats par saturation) →
  `make_candidate_crops.py` (montages numérotés) → classification vision → `ls_merge.py`.
- **Ré-annotation des manqués :** `extract_fp.py` (faux positifs d'un modèle entraîné sur un tamis
  tenu à l'écart — souvent de vraies particules non boxées) → montages → juge vision →
  `make_reannot_import.py`. **Sur TAMIS_B : +109 particules récupérées (74 % des « FP »),
  214 → 323 boîtes, précision 47 % → ~86 %.**
- **Enrichissement quasi-certain (reproductible) :** `enrich_extract.py` + `enrich_consolidate.py`
  ajoutent les candidats `conf ≥ 0.6` absents de la vérité-terrain. **+109 boîtes (847 → 956).**
- **Split par confiance (active learning) :**
  ```bash
  uv run python scripts/confidence_split.py data/corseacare_pred_tiled/counts.csv --high 0.6 --low 0.12
  ```
  → `ls_sure.json` (détections sûres, à juste **vérifier**) + `ls_borderline.json` (incertaines, à
  **corriger** avec pseudo-label). Une fois corrigée, la photo est considérée correctement annotée.

---

## 9. Couche Label Studio — snapshots portables + restore (FOCUS récent)

**Décision :** ne PAS committer `label_studio.sqlite3` brut (fuite email + hash mdp + token API,
+ chemins absolus non portables). À la place : **snapshots JSON portables + script de restauration**.

**Export** (`scripts/export_label_studio.py` → `make ls-export`) : lit le sqlite **directement**
(pas besoin du serveur ni d'un token), écrit un fichier par projet sous `labelstudio/` :
`{title, label_config, tasks:[{data, predictions, annotations}]}`. Les images sont référencées
`http://localhost:8081/corseacare/<file>` → **portables** (résolues par le serveur d'images après
`make annotate`). **Aucun secret exporté.**

Snapshots actuels (`labelstudio/index.json`) :

| fichier | tâches | prédictions | annotations | contenu |
|---|---:|---:|---:|---|
| `03_CorSeaCare.json` | 11 | 7 | 11 | jeu d'entraînement corrigé (= `annotations/`) |
| `04_CorSeaCare_preannot_verif.json` | 4 | 4 | 4 | vérification de pré-annotation |
| `05_csc_260606_1.json` | 27 | 27 | 0 | lot active-learning (pseudo-labels, en attente) |
| `08_csc_260606_2.json` | 21 | 21 | 0 | lot active-learning (en attente) |
| `09_csc_260606_3.json` | 21 | 21 | 0 | lot active-learning (en attente) |
| `10_csc_260606_4.json` | 21 | 21 | 1 | split par confiance (sure+borderline), **en cours** |

**Restore** (`scripts/restore_label_studio.py` → `make ls-restore`) : recrée chaque projet via
l'API (config + tâches + prédictions + annotations).
```bash
make annotate                       # Label Studio (:8080) + serveur d'images (:8081)
export LABEL_STUDIO_API_KEY=<PAT>
make ls-restore                      # tout ; ou: restore_label_studio.py --only CorSeaCare --skip-existing
```

**Auth (LS ≥ 1.23) — important :** le token *Access Token* legacy (`Token <key>`) est **désactivé**
par défaut. Les **Personal Access Tokens** sont des **JWT refresh** : le script les détecte (3
segments séparés par des points), les échange à `POST /api/token/refresh` → `access`, puis
utilise `Bearer <access>`. Fallback `Token` legacy conservé. **Round-trip vérifié en vrai** :
restore du snapshot 04 → 4 tâches / 4 annotations / 4 prédictions identiques, puis supprimé.

**Re-export après annotation :** `make ls-export` re-snapshote la base et on committe le diff.

---

## 10. Git & LFS — état de synchronisation

- **Working tree propre**, `main` = `origin/main`, **aucun commit non poussé**.
- `git lfs push origin main --dry-run` → **vide** (tous les objets LFS sont sur le remote).
- **LFS suivi :** **59 images** (`data/corseacare/`) + **2 poids** (`models/`) = 61 objets.
- `.gitattributes` : LFS sur `data/corseacare/*.jpg,*.JPG` et `models/*.pt`.
- `.gitignore` clés : `/data/*` avec négations `!/data/corseacare/*.jpg|JPG` ; `*.pt` global
  (donc SAM2 + poids de `runs/` ignorés) **mais** `!models/` + `!models/*.pt` ; `docs/superpowers/` ;
  sorties CLI racine.
- **Les snapshots Label Studio (`labelstudio/*.json`, ~11 Mo) sont en git NORMAL** (texte, diffable),
  **pas en LFS** — volontaire.

Un `git clone` + `git lfs pull` récupère **l'intégralité** (code + images + annotations + manifeste +
poids + couche LS) — entraînement ET inférence reproductibles ailleurs.

---

## 11. Cheat-sheet pratique

```bash
# --- Setup ---
git lfs install && git clone <url> && cd <repo> && git lfs pull && ./scripts/setup.sh
uv pip install "ultralytics>=8.3" "sam-2 @ git+https://github.com/facebookresearch/sam2.git"

# --- Inférence (recommandé) ---
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/predict_tiled.py models/corseacare_tiles_v7.pt 0.12

# --- Ré-entraîner depuis le dépôt ---
uv run python scripts/build_yolo_dataset.py && uv run python scripts/tile_dataset.py && uv run python scripts/train_tiles.py

# --- Mesure mm + rapport ---
uv run python scripts/calibrate_mesh.py calibrate && uv run python scripts/calibrate_mesh.py apply
uv run corseacare count data/corseacare --out counts.csv --tiled
uv run corseacare report --particles counts.csv

# --- Annotation / active learning ---
make annotate                                   # :8081 images + :8080 Label Studio
make ls-project TASKS=data/corseacare_preann/ls_tasks.json   # crée le projet (LABEL_STUDIO_API_KEY)
uv run python scripts/confidence_split.py data/corseacare_pred_tiled/counts.csv --high 0.6 --low 0.12
uv run python scripts/export_from_ls.py         # exporte tes annotations (dedupe par image)

# --- Couche Label Studio (snapshots) ---
make ls-export                                  # base LS -> labelstudio/*.json
make ls-restore                                 # labelstudio/*.json -> projets LS (PAT requis)

# --- Tests ---
uv run pytest
```

---

## 12. Pièges connus & corrections

| Symptôme | Cause | Correction |
|---|---|---|
| mAP « trop beau » (0,87) | `tile_dataset` ne vidait pas la sortie → fuite val→train | `rmtree` en tête de `tile_dataset.py` |
| mAP val bruité/non interprétable | MANTA étranger utilisé comme val | `split` explicite, val = TAMIS in-domain |
| Grandes boîtes rouges hors tamis | `roi_margin` trop lâche | `roi_margin=1.0` (jette au-delà du bord) |
| Plaques / boîtes géantes | pas de filtre de taille | `max_box_frac=0.04` |
| Petites particules manquées | seuil trop haut | `conf=0.12` à l'inférence |
| Particules **noires** manquées | proposeur saturation aveugle au noir + GT 4,4 % noir | TODO : proposeur sombre (§14) |
| `autre` affiché en orange dans LS | config sans couleur explicite | config canonique `autre background="#808080"` |
| Doublons après ré-import LS | LS n'écrase pas à l'import (empile) | supprimer les anciennes tâches d'abord |
| LS API 401 « legacy token disabled » | LS ≥ 1.23 | Personal Access Token (JWT) — échange refresh→access (§9) |

---

## 13. Sécurité

- **Aucun secret dans le dépôt** : pas de token, email, hash mdp, ni chemin absolu (vérifié par grep
  sur `labelstudio/`).
- La base `label_studio.sqlite3` **n'est jamais committée** (elle contient ces secrets).
- ⚠️ **Token partagé en chat** : le Personal Access Token fourni est un *refresh* à expiration
  quasi permanente (~an 2226). Il a été utilisé **uniquement en mémoire** pour le test (jamais écrit
  dans le dépôt). **Recommandation : le révoquer/régénérer** (LS → Account & Settings → Personal
  Access Token).
- Images publiées en **CC0** (responsabilité droits assumée, y compris MANTA étrangères ; consentement
  des personnes confirmé).

---

## 14. TODO / prochaines étapes

1. **Proposeur de candidats sombres** : seuillage des blobs sombres/haut-contraste *dans* le tamis
   (ROI + filtre taille), fusionné dans le lot borderline → faire remonter les particules noires
   manquées pour annotation. (Corrige le biais §3 / §12.)
2. **Boucle de finalisation** quand la révision de `csc_260606_4` est finie (sure + borderline +
   ajout des noirs) : `export_from_ls.py` → fusion sure + borderline corrigés → finaliser les
   annotations **TAMIS_E/F** → commit/push → ré-entraîner **tiles_v8** → comparer à v7.
3. Embarquer la note de téléchargement du checkpoint SAM2 si besoin (déjà dans README/`models/`).
```
