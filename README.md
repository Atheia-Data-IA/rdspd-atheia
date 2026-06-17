# rdspd-core — Simulateur de portefeuille de crédit synthétique (RDS PD)

`rdspd-core` est le refactoring en package Python du notebook **`SimulationRDSPD_V1.3`**.
Il génère un portefeuille synthétique de crédits revolving à la consommation, le fait
évoluer mois par mois sur une longue période, applique des **scorecards** de tirage
(revolving) et de **probabilité de défaut** (PD), puis produit les exports et les
contrôles de validation du modèle.

Le code est organisé pour que la **logique métier** soit totalement découplée des
entrées/sorties et de l'interface en ligne de commande, afin de pouvoir être appelée
ultérieurement par une **API (FastAPI)**. *La partie API et le frontend ne sont pas
développés ici.*

---

## 1. Ce que fait la simulation

| Étape | Description | Sortie principale |
|------|-------------|-------------------|
| **Octroi** | Génération de données emprunteurs corrélées non normales (générateur `IG`, système de Pearson), affectation des cohortes/années, dé-standardisation, variables catégorielles (CSP, habitat, situation familiale, ancienneté), dates. | `00_Granting.csv` |
| **Macro** | Import des facteurs macro-économiques : chômage et durées, inflation, indices de revenu par CSP, taux d'intérêt revolving, facteur macro agrégé. | — |
| **Scorecards** | Construction des grilles de score **RVV** (tirage revolving) et **PD** (défaut) par produit cartésien des modalités ; optimisation de la décomposition `PDt = P0 × P1 × P2`. | `ScoreGrid_RVV.csv`, `ScoreGrid_PD.csv` |
| **Simulation** | Boucle mensuelle (2006→2023) : mise à jour des drivers de risque, tirage du revolving, impayés et défaut conditionnels à P0/P1/P2 et à l'autocorrélation des actifs ρ, échéancier de recouvrement, clôture. Un pickle par mois. | `snapshots/*.pkl` |
| **Consolidation & contrôles** | Concaténation des snapshots, flag de défaut à 12 mois, contrôles de validation (covariance, distributions, intervalles de confiance sur P0-P2 / PDt / PD12). | `RDSPD_CompleteMaster&Fac.csv` |
| **Exports** | Jeu complet et snapshot performing de décembre. Archive des pickles. | `RDSPD_Full_<version>.csv`, `RDSPD_Perf_Dec_<version>.csv`, `archive_pickle.zip` |

---

## 2. Installation

Prérequis : **Python ≥ 3.10**.

```bash
# Depuis la racine du projet (rdspd-core/)
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell : .venv\Scripts\Activate.ps1)
# source .venv/bin/activate     # Linux / macOS

pip install -e .                # installation du package + dépendances
pip install -e ".[dev]"         # + outils de développement (pytest, ruff)
```

Dépendances runtime : `numpy`, `pandas`, `scipy`, `python-dateutil`, `tqdm`.

---

## 3. Données d'entrée

Les fichiers d'entrée sont lus **localement** depuis le dossier `inputs/` (et non plus
depuis Google Drive comme dans le notebook). L'arborescence attendue par défaut est :

```
inputs/01_Sujets_Internes/00_RDS_PD/Google_Sheets/
    RDSPD_CovarMatrix_Conso.csv        RDSPD_Quantile_Conso.csv
    RDSPD_Cohort_Revolving.csv         RDSPD_Unemployment.csv
    RDSPD_Unemployment_Duration.csv    RDSPD_Inflation.csv
    RDSPD_IncomeIndex.csv              RDSPD_InterestRate_Revolving.csv
    RDSPD_AggMacro.csv                 RDSPD_ProbaHousing_Conso.csv
    RDSPD_ProbaFamily_Conso.csv        RDSPD_ProbaCSP_Conso.csv
    RDSPD_ProbaEmploy_Conso.csv
    RDSPD_PartialScore_PD02R3120.csv   RDSPD_ScoreParams_PD02R3120.csv
    RDSPD_PartialScore_RVV00R0030.csv  RDSPD_ScoreParams_RVV00R0030.csv
```

> **Encodage** : certains fichiers (`RDSPD_ProbaCSP_Conso.csv`, `RDSPD_IncomeIndex.csv`,
> `RDSPD_ProbaHousing_Conso.csv`) sont encodés en **cp1252**. La lecture est robuste
> (UTF-8 puis cp1252/latin-1) afin que les libellés accentués (ex. « Indépendants »,
> « Propriétaire ») soient corrects — ils sont comparés en littéral par le scorecard PD.

Le répertoire d'entrée peut être changé avec `--inputs-dir`.

---

## 4. Exécution

```bash
# Run complet : 50 000 contrats, snapshots 2006→2023 (long, plusieurs dizaines de minutes)
rdspd-simulate

# Run rapide (smoke test) : N réduit, plage de dates courte
rdspd-simulate --smoke

# Run personnalisé
rdspd-simulate --n-contracts 5000 --end 2010-12-31 --seed 7 --rho 0.05
```

Options principales :

| Option | Défaut | Description |
|--------|--------|-------------|
| `--smoke` | — | Preset réduit (2 000 contrats, 2006→2007) |
| `--version` | `PD02R50k0623TTC` | Identifiant du run (= nom du dossier de sortie) |
| `--n-contracts` | `50000` | Taille du portefeuille |
| `--start` / `--end` | `2006-01-31` / `2023-12-31` | Plage de snapshots (fins de mois) |
| `--rho` | `0.05` | Autocorrélation des actifs (impact sur P0) |
| `--seed` | `42` | Graine aléatoire globale (reproductibilité) |
| `--inputs-dir` / `--outputs-dir` | `inputs/...` / `outputs/` | Répertoires d'E/S |
| `--scorecard-pd` / `--scorecard-rvv` | `PD02R3120` / `RVV00R0030` | Scorecards utilisées |

### Appel programmatique (futur usage API)

```python
from rdspd_core.config import SimulationConfig
from rdspd_core.pipeline import run

result = run(SimulationConfig(n_contracts=5000, end="2010-12-31"))
print(result.full_csv, result.n_rows_full)
```

---

## 5. Sorties

Toutes les sorties sont **affichées à l'exécution** *et* écrites dans
`outputs/<version>/` :

```
outputs/<version>/
    00_Granting.csv                      # portefeuille à l'octroi
    ScoreGrid_RVV.csv                    # grille de score revolving
    ScoreGrid_PD.csv                     # grille de score de défaut
    RDSPD_CompleteMaster&Fac.csv         # tous les snapshots consolidés
    RDSPD_Full_<version>.csv             # jeu complet final
    RDSPD_Perf_Dec_<version>.csv         # snapshot performing de décembre
    archive_pickle.zip                   # archive des pickles mensuels
    run.log                              # journal d'exécution
    controls/                            # rapports de contrôle (un CSV par contrôle)
    snapshots/                           # pickles mensuels (supprimés après archivage)
```

Les **contrôles de validation** (écart de covariance, distributions, distributions
conditionnelles, mapping drivers/score, intervalles de confiance P0-P2 / PDt / PD12)
sont imprimés sur la console et enregistrés dans `controls/`.

---

## 6. Reproductibilité

- Le notebook d'origine était **stochastique sans graine globale** : ses sorties variaient
  à chaque exécution. `rdspd-core` ajoute une **graine globale** (`--seed`) : deux runs avec
  la même configuration produisent des résultats identiques.
- Les **grilles de score** (`ScoreGrid_PD.csv`, `ScoreGrid_RVV.csv`) sont **déterministes**
  (produit cartésien + optimisation déterministe) : elles correspondent aux fichiers de
  référence. C'est l'ancre de vérification du portage (cf. tests).
- Les parties **granting** et **simulation** restent stochastiques : à graine fixée elles
  sont reproductibles, mais ne reproduisent pas bit-à-bit la réalisation particulière du
  notebook (qui n'avait pas de graine). Les **fichiers, colonnes et contrôles** sont
  néanmoins identiques et statistiquement équivalents.

---

## 7. Tests

```bash
pytest                              # tous les tests
pytest tests/test_scores_deterministic.py   # grilles == références
```

- `test_scores_deterministic.py` : les grilles RVV/PD générées correspondent aux références.
- `test_encoding.py` : les libellés accentués (CSP, habitat) sont correctement décodés.
- `test_smoke_pipeline.py` : run réduit de bout-en-bout, vérification des exports.

---

## 8. Architecture

```
src/rdspd_core/
    config.py        # SimulationConfig : toute la paramétrisation
    io_utils.py      # lecture CSV robuste (encodage), écriture
    reporting.py     # Reporter : affichage console + écriture fichiers
    randnonnormal/   # générateur IG (système de Pearson) vendorisé
    granting.py      # octroi des contrats
    macro.py         # facteurs macro-économiques
    risk_drivers.py  # mise à jour des drivers de risque
    scores.py        # grilles de score RVV/PD + application
    snapshots.py     # cycle de vie contrats/facilités + pickles
    simulation.py    # boucle mensuelle
    controls.py      # contrôles de validation
    pipeline.py      # orchestration bout-en-bout (point d'entrée API)
    cli.py           # interface ligne de commande
```

---

## 9. Notes & limites

- **Performance** : le run complet (50 000 contrats × 216 mois) prend plusieurs dizaines de
  minutes (boucle mensuelle + E/S pickle). Utiliser `--smoke` ou réduire `--n-contracts` /
  la plage de dates pour des essais rapides.
- **Fidélité** : la logique métier et l'ordre des tirages aléatoires sont préservés à
  l'identique du notebook ; seules des améliorations sans effet sur les résultats ont été
  apportées (suppression d'effets de bord, lecture d'encodage, ordre déterministe des
  classes d'âge, noms de pickles compatibles Windows).
- **API / Frontend** : hors périmètre de ce dépôt. La fonction `pipeline.run(config)` est le
  point d'entrée prévu pour une future couche API.
