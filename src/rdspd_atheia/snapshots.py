"""Gestion des snapshots mensuels : pickles, initialisation et mise a jour.

Reproduit les cellules « Pickles tools » et « Snapshot update » du notebook. Un snapshot
mensuel comprend deux types de lignes :

- les **contrats** (master) : un enregistrement par contrat revolving ;
- les **facilites** (sub) : un enregistrement par tirage (utilisation du credit).

NOTE Windows : le notebook nommait les pickles ``Snapshot_{Timestamp}.pkl`` ; l'horodatage
brut contient ``:`` (illegal sous Windows). On utilise donc ``Snapshot_YYYY-MM-DD.pkl``
(les dates sont des fins de mois, ce format est sans ambiguite).
"""

from __future__ import annotations

import glob
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .io_utils import read_csv
from .macro import MacroData
from .risk_drivers import (
    increment_age,
    increment_aggregates,
    increment_banking_seniority,
    increment_dates,
    increment_empsen,
    increment_income_wages,
    increment_unemp,
)
from .scores import (
    PD_KEYLIST,
    PD_VARLIST,
    RVV_KEYLIST,
    RVV_VARLIST,
    pdscore_apply,
    pdscore_rd,
    rvvscore_apply,
    rvvscore_rd,
)


# ---------------------------------------------------------------------------
# Outils pickle
# ---------------------------------------------------------------------------
def _snapshot_filename(dt: pd.Timestamp) -> str:
    """Nom de fichier pickle (sans ambiguite et compatible Windows)."""
    return f"Snapshot_{pd.Timestamp(dt).strftime('%Y-%m-%d')}.pkl"


def snap_pickle_import(dt: pd.Timestamp, snapshots_dir: str | Path, verbose: bool = False):
    """Charge le snapshot d'une date donnee, ou ``None`` s'il n'existe pas."""
    input_file = Path(snapshots_dir) / _snapshot_filename(dt)
    if input_file.exists():
        with open(input_file, "rb") as f:
            out_snap = pickle.load(f)
            if verbose:
                print("Pickle import OK")
        return out_snap
    return None


def snap_pickle_export(
    dt: pd.Timestamp,
    snpctr: pd.DataFrame,
    snpfac: pd.DataFrame,
    snapshots_dir: str | Path,
    verbose: bool = False,
) -> None:
    """Concatene contrats + facilites et serialise le snapshot du mois."""
    snap = pd.concat([snpctr, snpfac], ignore_index=False)
    output_file = Path(snapshots_dir) / _snapshot_filename(dt)
    with open(output_file, "wb") as f:
        pickle.dump(snap, f)
        if verbose:
            print("Pickle creation OK")


def pickle_mass_import(snapshots_dir: str | Path) -> pd.DataFrame:
    """Concatene tous les pickles de snapshots du repertoire."""
    pkllist = glob.glob(os.path.join(str(snapshots_dir), "*.pkl"))
    return pd.concat([pd.read_pickle(f) for f in pkllist], ignore_index=False)


def tools_del_pkl(snapshots_dir: str | Path, verbose: bool = True) -> None:
    """Supprime tous les pickles du repertoire."""
    path = str(snapshots_dir)
    for file in glob.glob(os.path.join(path, "*.pkl")) + glob.glob(os.path.join(path, "*.pickle")):
        os.remove(file)
        if verbose:
            print(f"Deleted file: {file}")


def pickle_mass_zip(snapshots_dir: str | Path, archive_path: str | Path, verbose: bool = True) -> None:
    """Archive tous les pickles dans un zip puis les supprime."""
    snapshots_dir = Path(snapshots_dir)
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in os.listdir(snapshots_dir):
            if file.endswith(".pkl"):
                archive.write(snapshots_dir / file, arcname=file)
    if verbose:
        print(f"Archive successfully created into: {archive_path}")
    tools_del_pkl(snapshots_dir, verbose=False)
    if verbose:
        print("    > All pickle files deleted")


def pickle_mass_unzip(snapshots_dir: str | Path, archive_path: str | Path, verbose: bool = True) -> None:
    """Decompresse l'archive de pickles puis la supprime."""
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(snapshots_dir)
    if verbose:
        print(f"Pickles unzip into: {snapshots_dir}")
    os.remove(archive_path)
    if verbose:
        print("    > Archive file deleted")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_grid(cfg: SimulationConfig, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relit les grilles de score exportees (round-trip CSV indispensable, cf. module ``scores``)."""
    score_grid_pd = read_csv(cfg.output_path("ScoreGrid_PD.csv"))
    if verbose:
        print("ScoreGrid_PD uploaded OK")
    score_grid_rvv = read_csv(cfg.output_path("ScoreGrid_RVV.csv"))
    if verbose:
        print("ScoreGrid_RVV uploaded OK")
    return score_grid_pd, score_grid_rvv


def init_granting(cfg: SimulationConfig, verbose: bool = False) -> pd.DataFrame:
    """Relit ``00_Granting.csv`` et initialise toutes les colonnes techniques."""
    data = read_csv(cfg.output_path("00_Granting.csv"))
    data.set_index("Contract_ID", inplace=True)

    # Identifiant de facilite + statut
    data["Facility_ID"] = 0
    data["Master"] = True
    data["Drawing"] = 1

    # Conversion des dates
    data["Date"] = pd.to_datetime(data["Date"])
    data["Birth_dt"] = pd.to_datetime(data["Birth_dt"])
    data["Employment_dt"] = pd.to_datetime(data["Employment_dt"])
    data["Banking_dt"] = pd.to_datetime(data["Banking_dt"])

    # Recalcul age et anciennete d'emploi
    data["Age"] = ((data.Date - data.Birth_dt).dt.days / 365.25).astype(int)
    data["anc_emp_continuous"] = ((data.Date - data.Employment_dt).dt.days / 365.25).astype(int)

    # Autres dates
    data["Creation_dt"] = data["Date"]
    data["Creation_yr"] = data["Annee"]
    data["Closure_dt"] = pd.NaT

    # Autres drivers de risque (montants en centimes)
    data["Charges_ajuste"] = (data["Charges"] * 100).astype(int)
    data["Revenu_ajuste"] = (data["Revenu"] * 100).astype(int)

    # Initialisation des indicateurs de snapshot
    data[
        [
            "ScorePD", "PD12", "PDt", "P0", "P1", "P2",
            "Retard", "Retard_cumule", "Defaut",
            "Unemp", "Duration",
            "dec_flag", "dispo_ratio", "dispo_ratio_bin", "draw_senior", "draw_senior_bin",
            "ScoreRVV", "RVV12", "RVVt", "Drawing_override",
            "Inactive", "Inactive_senior", "Closed",
        ]
    ] = 0

    # Parametres credit
    data["Disposable"] = data["plafond_credit"]

    # Agregats
    data[["NbFac", "ResidualMaturity"]] = 0

    # Etats financiers
    data[
        [
            "Principal", "Outstanding", "performing_claim", "montant_credit",
            "Claim_PD_Principal", "Claim_PD_Interest", "Claim_PD_Fees", "Claim_Interest", "Claim_Principal", "Claim_Total",
            "Recov_Total", "Recov_Interest", "Recov_Principal", "Recov_PD_Interest", "Recov_PD_Fees", "Recov_PD_Principal", "Recov_Overdue",
            "PastDue_Principal", "PastDue_Interest", "PastDue_Fees", "PastDue_Cap_Interest", "PastDue_Cap_Fees", "PastDue_Total",
        ]
    ] = 0

    if verbose:
        print("")
        print("Variables initialization OK")
        print("   Number of facilities: ", len(data))

    return data


def init_snapshot_ctr(data: pd.DataFrame, dt: pd.Timestamp, verbose: bool = False) -> pd.DataFrame:
    """Initialise le premier snapshot : nouveaux contrats octroyes a la date ``dt``."""
    new = data.query("Date == @dt")
    if verbose:
        print("Snapshot initialization OK")
        print("   Snapshot: ", dt.strftime("%y/%m/%d"))
        print("   New facs: ", len(new))
    return new


def append_snapshot_ctr(updt: pd.DataFrame, new: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Concatene contrats mis a jour + nouveaux contrats octroyes."""
    snap = pd.concat([updt, new], ignore_index=False)
    if verbose:
        print("Appending updated snapshot and new originations OK")
        print("   Total contracts: ", len(snap))
    return snap


# ---------------------------------------------------------------------------
# Mise a jour des contrats
# ---------------------------------------------------------------------------
def increment_snapshot_ctr(
    data: pd.DataFrame, dt: pd.Timestamp, last: pd.DataFrame, verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclut les contrats clos/en defaut du snapshot precedent et separe ctr/fac.

    Optimisation (neutre) : le snapshot precedent ``last`` est passe en memoire par la boucle
    au lieu d'etre relu depuis le pickle. Le pickle etant un round-trip sans perte, l'objet en
    memoire est identique a sa relecture ; les pickles restent ecrits pour la consolidation finale.
    """
    updt = last[last.Closed == 0]
    updt = updt[updt.Defaut == 0]

    updtctr = updt[updt.Master]
    updtfac = updt[updt.Master == False]  # noqa: E712 - reproduit le notebook

    newctr = data.query("Date == @dt")

    if verbose:
        print("Snapshot increment OK")
        print("   Snapshot: ", dt.strftime("%y/%m/%d"))
        print("   In portfolio: ", len(updtctr), " contracts")
        print("       ", len(updtfac), " facilities")
        print("   Closed contracts: ", last[last.Master].Closed.sum())
        print("   Defaulted contracts: ", last[last.Master].Defaut.sum())
        print("   New contracts: ", len(newctr))

    return updtctr, newctr, updtfac


def update_snapshot_ctr(
    updt: pd.DataFrame, dt: pd.Timestamp, macro: MacroData, score_grid_rvv: pd.DataFrame
) -> pd.DataFrame:
    """Met a jour les drivers de risque d'un contrat et applique le score revolving."""
    updt = increment_dates(updt, dt)
    updt = increment_unemp(updt, dt, macro.unemp_rates, macro.duration, macro.duration_proba)
    updt = increment_income_wages(updt, macro.inflation_df, macro.income_df)
    updt = increment_age(updt)
    updt = increment_empsen(updt)
    updt = increment_banking_seniority(updt)

    updt = rvvscore_rd(updt)
    updt = rvvscore_apply(updt, score_grid_rvv, RVV_KEYLIST, RVV_VARLIST)

    return updt


def update_default_ctr(
    master: pd.DataFrame,
    sub: pd.DataFrame,
    score_grid_pd: pd.DataFrame,
    rho: float,
    aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrege les encours, applique le score PD et propage impayes/defaut aux facilites."""
    master = increment_aggregates(master, sub)
    master = pdscore_rd(master)
    master = pdscore_apply(master, score_grid_pd, PD_KEYLIST, PD_VARLIST, rho=rho, aggregate=aggregate)
    # Optimisation (neutre) : remplace sub.update(master[...]). Propage la valeur du contrat a
    # chacune de ses facilites (index duplique) ; ``reindex`` diffuse par Contract_ID et l'on
    # conserve la valeur d'origine la ou le contrat est absent (NaN), comme le fait update().
    for col in ["Retard", "Retard_cumule", "Defaut"]:
        src = master[col].reindex(sub.index)
        updated = np.where(src.notna().to_numpy(), src.to_numpy(), sub[col].to_numpy())
        sub[col] = updated.astype(sub[col].dtype)
    return master, sub


def increment_closure_ctr(df: pd.DataFrame) -> pd.DataFrame:
    """Determine l'inactivite et la cloture d'un contrat (12 mois inactifs)."""
    df["Inactive"] = np.where(df.Outstanding + df.Retard == 0, 1, 0)
    df["Inactive_senior"] = (df.Inactive_senior * df.Inactive) + df.Inactive
    df["Closed"] = np.where(df.Inactive_senior == 12, 1, 0)
    df["Closure_dt"] = np.where(df.Closed == 1, df.Date, df.Closure_dt)
    return df


def update_financials_ctr(master: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Consolide les etats financiers des facilites au niveau du contrat + cloture."""
    varlist = [
        "Claim_PD_Principal", "Claim_PD_Interest", "Claim_PD_Fees", "Claim_Interest", "Claim_Principal", "Claim_Total",
        "Recov_Total", "Recov_Interest", "Recov_Principal", "Recov_PD_Interest", "Recov_PD_Fees", "Recov_PD_Principal", "Recov_Overdue",
        "PastDue_Principal", "PastDue_Interest", "PastDue_Fees", "PastDue_Cap_Interest", "PastDue_Cap_Fees", "PastDue_Total",
        "Principal", "Outstanding", "montant_credit", "performing_claim",
    ]
    df = sub.groupby("Contract_ID")[varlist].sum()
    # Optimisation (neutre) : remplace master.update(df) (df sans NaN, index inclus dans master).
    idx = master.index.intersection(df.index)
    for col in varlist:
        master.loc[idx, col] = df.loc[idx, col]
    master["Disposable"] = np.maximum(0, master["plafond_credit"] - master["Outstanding"])
    master = increment_closure_ctr(master)
    return master


# ---------------------------------------------------------------------------
# Facilites (tirages)
# ---------------------------------------------------------------------------
def origin_fac(master: pd.DataFrame, sub: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Cree de nouvelles facilites pour les contrats ayant tire (Drawing==1)."""
    df = master.query("Drawing==1")

    if len(sub) > 0:
        _nbfac = sub.groupby("Contract_ID")["Facility_ID"].max()
        # Optimisation (neutre) : remplace df.update(_nbfac) (Serie sans NaN) par une affectation
        # alignee sur l'intersection des index.
        idx = df.index.intersection(_nbfac.index)
        df.loc[idx, "Facility_ID"] = _nbfac.loc[idx]

    df["Facility_ID"] += 1
    df["Master"] = False
    df["Drawing"] = 0

    if verbose:
        print("New drawing origination OK")
        print("   New facilities: ", len(master.query("Drawing==1 and Creation_dt<Date")))

    return df


def init_snapshot_fac(df: pd.DataFrame, dt: pd.Timestamp, interest_rates: pd.DataFrame) -> pd.DataFrame:
    """Initialise les nouvelles facilites : montant tire, taux, maturite, echeancier."""
    df["Creation_dt"] = pd.to_datetime(dt)

    # Montant tire (arrondi a 5000 centimes)
    df["montant_credit"] = (
        np.random.uniform(low=15000, high=df["Disposable"], size=len(df)) / 5000
    ).astype(int) * 5000
    df["Principal"] = df.montant_credit
    df["Outstanding"] = df.montant_credit
    df["Disposable"] = 0

    # Taux d'interet
    if "InterestRateAvg" in df.columns:
        del df["InterestRateAvg"]
    df = df.merge(interest_rates, on="Date", how="left").set_index(df.index)
    df["interest_rate_month"] = df.InterestRateAvg / 12

    # Maturite (selon le montant)
    conditions = [
        df.montant_credit < 50000,
        df.montant_credit < 300000,
    ]
    distrib = [
        (np.random.randint(6, 25, size=len(df)) / 6).astype(int) * 6,
        (np.random.randint(12, 37, size=len(df)) / 12).astype(int) * 12,
    ]
    default = (np.random.randint(36, 61, size=len(df)) / 12).astype(int) * 12
    df["Maturite"] = np.select(conditions, distrib, default=default)
    df["ResidualMaturity"] = df.Maturite

    # Mensualite constante (echeancier d'amortissement)
    df["performing_claim"] = np.ceil(
        df.interest_rate_month / (1 - (1 + df.interest_rate_month) ** (-df.Maturite)) * df.montant_credit
    ).astype(int)

    # Etats financiers initiaux
    df[
        [
            "Claim_PD_Principal", "Claim_PD_Interest", "Claim_PD_Fees", "Claim_Interest", "Claim_Principal", "Claim_Total",
            "Recov_Total", "Recov_Interest", "Recov_Principal", "Recov_PD_Interest", "Recov_PD_Fees", "Recov_PD_Principal", "Recov_Overdue",
            "PastDue_Principal", "PastDue_Interest", "PastDue_Fees", "PastDue_Cap_Interest", "PastDue_Cap_Fees", "PastDue_Total",
        ]
    ] = 0

    return df


def append_snapshot_fac(updt: pd.DataFrame, new: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Concatene facilites mises a jour + nouvelles facilites."""
    snap = pd.concat([updt, new], ignore_index=False)
    if verbose:
        print("Appending updated facilities and new facilities OK")
        print("   Total facilities: ", len(snap))
    return snap


def update_snapshot_fac(updt: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    """Met a jour les dates et la maturite residuelle des facilites."""
    updt = increment_dates(updt, dt)
    updt["Creation_dt"] = pd.to_datetime(updt.Creation_dt)
    updt["ResidualMaturity"] = (
        updt.Maturite
        - (updt.Date.dt.year - updt.Creation_dt.dt.year) * 12
        - (updt.Date.dt.month - updt.Creation_dt.dt.month)
    )
    updt["ResidualMaturity"] = np.where((updt.ResidualMaturity < 1) & (updt.Retard > 0), 1, updt.ResidualMaturity)
    updt["Disposable"] = 0
    return updt


def increment_recoveryschedule(snap: pd.DataFrame, fees: int = 200) -> pd.DataFrame:
    """Calcule l'echeancier de recouvrement, les recouvrements et les impayes du mois."""
    last = snap.copy()
    # Creances issues des impayes anterieurs
    snap["Claim_PD_Principal"] = last.PastDue_Principal
    snap["Claim_PD_Interest"] = last.PastDue_Cap_Interest
    snap["Claim_PD_Fees"] = last.PastDue_Cap_Fees

    # Creances de l'appel courant
    snap["Claim_Interest"] = round((last.Principal + snap.Claim_PD_Principal) * snap.interest_rate_month).astype(int)
    snap["Claim_Principal"] = np.minimum(snap.performing_claim - snap.Claim_Interest, last.Principal).astype(int)
    snap["Claim_Total"] = (
        snap.Claim_Interest + snap.Claim_Principal + snap.Claim_PD_Principal + snap.Claim_PD_Interest + snap.Claim_PD_Fees
    ).astype(int)

    # Recouvrements par ordre de priorite
    snap["Recov_Total"] = snap.Claim_Total * (1 - snap.Retard)
    snap["_toberecov"] = snap.Recov_Total
    snap["Recov_Interest"] = np.minimum(snap.Claim_Interest, snap._toberecov)
    snap["_toberecov"] -= snap.Recov_Interest
    snap["Recov_Principal"] = np.minimum(snap.Claim_Principal, snap._toberecov)
    snap["_toberecov"] -= snap.Recov_Principal
    snap["Recov_PD_Interest"] = np.minimum(snap.Claim_PD_Interest, snap._toberecov)
    snap["_toberecov"] -= snap.Recov_PD_Interest
    snap["Recov_PD_Fees"] = np.minimum(snap.Claim_PD_Fees, snap._toberecov)
    snap["_toberecov"] -= snap.Recov_PD_Fees
    snap["Recov_PD_Principal"] = np.minimum(snap.Claim_PD_Principal, snap._toberecov)

    # Impayes (Past Due)
    snap["PastDue_Principal"] = (snap.Claim_PD_Principal + snap.Claim_Principal) - (snap.Recov_Principal + snap.Recov_PD_Principal)
    snap["PastDue_Interest"] = snap.Claim_Interest - snap.Recov_Interest
    snap["PastDue_Fees"] = np.where((snap.PastDue_Principal + snap.PastDue_Interest) > 0, fees, 0)
    snap["PastDue_Cap_Interest"] = (snap.Claim_PD_Interest + snap.Claim_Interest) - (snap.Recov_Interest + snap.Recov_PD_Interest)
    snap["PastDue_Cap_Fees"] = snap.Claim_PD_Fees - snap.Recov_PD_Fees + snap.PastDue_Fees
    snap["PastDue_Total"] = snap.PastDue_Principal + snap.PastDue_Cap_Interest + snap.PastDue_Cap_Fees

    # Fin de snapshot
    snap["Principal"] = last.Principal - snap.Claim_Principal
    snap["Outstanding"] = snap.Principal + snap.PastDue_Total

    del snap["_toberecov"]

    return snap


def increment_closure_fac(df: pd.DataFrame) -> pd.DataFrame:
    """Determine la cloture d'une facilite (encours nul)."""
    df["Inactive"] = 0
    df["Inactive_senior"] = 0
    df["Closed"] = np.where(df.Outstanding == 0, 1, 0)
    df["Closure_dt"] = np.where(df.Closed == 1, df.Date, df.Closure_dt)
    return df


def update_financials_all(master: pd.DataFrame, sub: pd.DataFrame, fees: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Met a jour les etats financiers des facilites puis consolide au niveau contrat."""
    sub = increment_recoveryschedule(sub, fees=fees)
    sub = increment_closure_fac(sub)
    master = update_financials_ctr(master, sub)
    return master, sub
