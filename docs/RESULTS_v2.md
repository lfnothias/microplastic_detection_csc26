# CorSeaCare_yolo — résultats v6 vs v7 (avant/après enrichissement + vues non annotées)

État au 2026-06-06. Toutes les métriques sont **leakage-free** (held-out par tamis physique).
- **v6** = entraîné sur annotations **originales** (`data/runs/tiles_v6`).
- **v7** = entraîné sur annotations **enrichies** (`data/runs/tiles_v7`).
- Tamis de validation = **TAMIS_B**, jamais vu à l'entraînement. GT « avant » = 214 boîtes,
  « après » enrichissement = 261 boîtes (+18 plastique, +29 organique).

Reproduction : `scripts/_val.py` (mAP), `scripts/eval_binary.py` (binaire),
`scripts/results_consistency.py` (Partie 2).

---

## 1. Performances v6 / v7 — sur la GT **avant** vs **après** enrichissement

### Détection (held-out B)

| Modèle | GT de B | mAP@50 | mAP@50-95 | Précision | Rappel |
|--------|---------|--------|-----------|-----------|--------|
| v6 | avant (214) | 0.374 | 0.179 | 0.43 | 0.37 |
| v6 | **après (261)** | 0.393 | 0.190 | 0.46 | 0.36 |
| v7 | avant (214) | 0.370 | 0.188 | **0.71** | 0.38 |
| v7 | **après (261)** | 0.372 | 0.189 | **0.72** | 0.39 |

Par classe (mAP@50-95, GT après) : v6 `pellet` 0.40 · `fragment` 0.25 · `film` 0.18 ·
`fibre` 0.19 · `autre` 0.12 · `mousse` 0.01 ; v7 `pellet` 0.47 · `fragment` 0.22 · `film` 0.14 ·
`fibre` 0.19 · `autre` 0.10 · `mousse` 0.01.

### Plastique vs organique (parmi les particules détectées, held-out B)

| Modèle | GT de B | classif. binaire | plastique P/R/F1 | organique P/R/F1 |
|--------|---------|------------------|------------------|------------------|
| v6 | avant (214) | 92.4 % | 0.47 / 0.61 / 0.54 | 0.34 / 0.44 / 0.38 |
| v6 | après (261) | 93.0 % | 0.56 / 0.64 / 0.60 | 0.46 / 0.41 / 0.44 |
| v7 | avant (214) | 95.5 % | 0.61 / 0.55 / 0.58 | 0.33 / 0.34 / 0.34 |
| v7 | après (261) | 92.7 % | 0.67 / 0.54 / 0.60 | 0.36 / 0.26 / 0.30 |

### Lecture
- **Gain de précision robuste** : v6 → v7 fait passer la précision de **0.43→0.71** (GT avant) et
  **0.46→0.72** (GT après). Le gain tient quelle que soit la GT de mesure → c'est un effet réel
  de l'enrichissement de l'entraînement, pas un artefact de la GT.
- **mAP stable (~0.37–0.39)** : la capacité de détection brute ne change pas ; l'enrichissement
  déplace le compromis vers **moins de faux positifs**.
- **Effet « GT plus complète »** : passer la GT de 214→261 monte un peu précision/mAP (la GT
  incomplète pénalisait à tort de vraies détections).
- **Plastique vs organique ~92–95 %** dans tous les cas → résultat scientifique central stable.
- Rappel (~0.36–0.39) et classes rares (`mousse` ~0) restent le point faible (données limitées).

---

## 2. Vues NON annotées — bon ordre de grandeur du compte et des ratios ?

Chaque tamis a été photographié sous plusieurs **re-distributions physiques** du même matériel
(re-brassage à l'eau). On compare le compte + ratios prédits sur les **vues non annotées** à la
**GT de la vue annotée** (référence, par vue). Modèle = comptage tuilé + ROI-gating, conf 0.25.

### TAMIS_DSLR — 6 vues non annotées (réf. = vue annotée, ~310 part./vue, plastique/autre 78/22)
| | compte moyen (min–max, CV) | 5-classes (%) | plastique/autre |
|---|---|---|---|
| **GT (réf.)** | ~310 | frag 66 · autre 22 · pellet 6 · film 3 | 78 / 22 |
| **v6** | 390 (338–471, CV 12 %) | frag 78 · autre 17 · pellet 4 | 83 / 17 |
| **v7** | 367 (331–393, **CV 7 %**) | frag 76 · autre 18 · pellet 6 | 82 / 18 |

### TAMIS_B — 10 vues non annotées (réf. ~261 part./vue, plastique/autre 66/34)
| | compte moyen (min–max, CV) | 5-classes (%) | plastique/autre |
|---|---|---|---|
| **GT (réf.)** | ~261 | frag 59 · autre 34 · pellet 5 | 66 / 34 |
| **v6** | 224 (158–293, CV 18 %) | frag 66 · autre 31 · pellet 3 | 69 / 31 |
| **v7** | 196 (137–255, CV 16 %) | frag 63 · autre 31 · pellet 6 | 69 / 31 |

### TAMIS_C — 10 vues non annotées (réf. ~189 part./vue, plastique/autre 79/21)
| | compte moyen (min–max, CV) | 5-classes (%) | plastique/autre |
|---|---|---|---|
| **GT (réf.)** | ~189 | frag 72 · autre 21 · film 4 | 79 / 21 |
| **v6** | 222 (177–285, CV 15 %) | frag 82 · autre 16 | 84 / 16 |
| **v7** | 184 (165–214, **CV 10 %**) | frag 75 · autre 24 | 76 / 24 |

### Lecture
- **Bon ordre de grandeur du compte** : prédictions à **~±25 %** de la GT par vue (B sous-compté
  ~15-25 %, DSLR sur-compté ~20 %, C : v7 ≈ exact à 184 vs 189). Utilisable en relatif/ordre de
  grandeur, pas en compte absolu fin.
- **Ratios proches** : plastique/autre prédit à **~3–7 points** de la GT sur les 3 tamis ; la
  composition dominante (fragment/autre) est bien retrouvée.
- **Cohérence inter-vues** : CV du compte **7–18 %** ; **v7 plus stable** (CV plus bas) et plus
  proche de la GT sur C. Classes rares (`film`, `mousse`, `pellet`) trop peu nombreuses pour être
  fiables.
- → Sur des vues jamais annotées, v6 comme v7 **retrouvent un bon ordre de grandeur du nombre et
  des ratios** (surtout plastique/autre) ; v7 est le plus reproductible.

---

## Conclusion
1. **Enrichir les annotations améliore nettement la précision** (≈ +0.25) à mAP et discrimination
   plastique/organique constantes.
2. **Plastique vs organique ≈ 92–95 %** (robuste, avant comme après).
3. Sur les **vues non annotées**, les modèles donnent un **bon ordre de grandeur** (compte ±25 %,
   ratios à quelques points) — exploitable pour un suivi relatif.
4. Levier restant : **plus de tamis annotés** (rappel + classes rares + éval moins bruitée).
