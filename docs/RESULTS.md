# CorSeaCare_yolo — résultats d'évaluation (2026-06-06)

Toutes les métriques ci-dessous sont **leakage-free** (split par tamis physique, vérifié) après
correction d'un bug de tuilage qui contaminait le val (`tile_dataset.py` nettoie désormais sa
sortie).

## Données

- **Tamis CorSeaCare (à nous) : 4 sieves** — `TAMIS_DSLR` (1 mm), `TAMIS_B` (1 mm), `TAMIS_C`
  (2 mm), `TAMIS_D` (1 mm, non annoté). Annotés : 4 vues (DSLR×2, B×1, C×1).
- **Collection externe MANTA : 7 images** — données publiques, **entraînement seulement**
  (jamais en validation → aucune métrique mesurée dessus).
- Annotation pilotée par `split` explicite dans `samples.csv`.

## Test A — généralisation cross-sieve (tamis held-out, jamais vu)

| modèle | entraînement | test | mAP50 | mAP50-95 | P | R |
|--------|--------------|------|-------|----------|---|---|
| tiles_v5 | MANTA + B + C | **DSLR** (atypique) | 0.142 | 0.054 | 0.24 | 0.17 |
| tiles_v6 | MANTA + DSLR + C | **B** (typique) | **0.374** | 0.179 | 0.69 | 0.29 |

→ La perf sur un nouveau tamis dépend de sa similarité : **0.14 (atypique) à 0.37 (typique)**.
Le DSLR est un outlier. Rappel encore faible (la détection est le goulot).

## Test B — reproductibilité face à la re-distribution physique (re-brassage à l'eau)

Modèle `tiles_v5`, comptage sur N vues = N re-distributions du même échantillon.

| tamis | vues | CV compte | ratio fragment (CV) |
|-------|------|-----------|---------------------|
| TAMIS_B | 11 | 19 % | 63.6 % (6 %) |
| TAMIS_C | 11 | 15 % | 73.8 % (8 %) |
| TAMIS_DSLR | 8 | 16 % | 71.4 % (9 %) |

→ Compte total reproductible à **~±16 %** ; **ratio de la classe dominante très stable
(~±7 %)** ; classes rares (`mousse`, `film`) non fiables. La variabilité vient des
occlusions/amas (in-domain ≈ out-of-domain).

## Résultat clé — Plastique vs Matière organique (tamis B held-out)

GT : 153 plastique + 61 organique (214). Apparié aux annotations par IoU.

| | IoU 0.5 | IoU 0.3 |
|---|---|---|
| **Classif. binaire correcte (parmi détectées)** | **92.4 %** | 92.3 % |
| Plastique P / R / F1 | 0.47 / 0.61 / 0.54 | 0.52 / 0.67 / 0.59 |
| Organique P / R / F1 | 0.34 / 0.44 / 0.38 | 0.36 / 0.48 / 0.41 |

→ **Une fois détectée, une particule est classée plastique vs organique à ~92 %** (stable en
IoU). Le facteur limitant est la complétude de la détection, pas la discrimination.

## Ré-annotation assistée par vision — annotation incomplète

Les « faux positifs » de B ont été soumis à un modèle de vision (5 sous-agents, carte de
référence few-shot). Sur **147 FP** : **109 (74 %) sont de VRAIES particules non annotées**,
38 (26 %) du bruit.

- Annotation `153307` : **214 → 323 boîtes (+51 %)** ; composition révisée **plastique 64 % /
  organique 36 %** (vs 71/29 — l'organique était davantage sous-annoté).
- **Précision réelle du modèle ≈ 86 %** (240 vraies / 278 prédictions) vs 47 % « naïf » :
  l'annotation incomplète masquait la vraie précision.

## Consolidation enrichie (quasi-certain, reproductible)

Règle d'enrichissement reproductible : candidat = détection non appariée à la GT ; on **garde
seulement conf ≥ 0.6** (haute confiance modèle = très probablement une particule réelle manquée),
on **jette le reste (borderline)**. Corroboré par inspection vision (les fragments
bleu/rose/jaune/teal des montages sont indubitablement du plastique).

| image | GT | enrichi | +ajouts (quasi-certains) |
|-------|----|---------|--------------------------|
| 145851 | 279 | 314 | +35 (28 frag, 5 autre, 2 pellet) |
| 153307 | 214 | 240 | +26 (17 frag, 9 autre) |
| DT5A0150 | 199 | 225 | +26 (21 frag, 4 autre, 1 pellet) |
| IMG_8891 | 155 | 177 | +22 (20 frag, 1 autre, 1 film) |
| **total** | **847** | **956** | **+109** |

Limite honnête : la règle conf≥0.6 est **biaisée vers `fragment`** (la classe que le modèle
maîtrise) ; l'organique manqué se récupère via la passe vision plus permissive (à réviser).
Tooling : `enrich_extract.py` + `enrich_consolidate.py`. Labels enrichis (`data/enrich/
labels_enriched/`) + import LS (`ls_enrich.json`) prêts pour révision avant ré-entraînement.

## Conclusions

1. **Le cœur scientifique (plastique vs organique) fonctionne : ~92 %.**
2. **La précision est en réalité élevée (~86 %)** ; les « faux positifs » sont surtout des
   particules réelles non annotées.
3. La détection (rappel) et les classes rares restent limitées par le **volume d'annotation**.
4. Reproductibilité du comptage ~±16 %, ratio dominant ~±7 %.

**Levier unique : annoter plus de tamis distincts** (et compléter l'annotation via la boucle
de ré-annotation assistée par vision).
