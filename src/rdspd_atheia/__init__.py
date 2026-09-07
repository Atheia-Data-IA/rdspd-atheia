"""rdspd_atheia - Simulateur de portefeuille de credit synthetique RDS PD.

Ce package est le refactoring du notebook ``SimulationRDSPD_V1.3`` en code Python
modulaire et executable. La logique metier (generation des contrats, scores,
simulation mensuelle) est decouplee des entrees/sorties et de l'interface en ligne
de commande, afin de pouvoir etre appelee ulterieurement par une API (FastAPI).

Point d'entree principal :
    from rdspd_atheia.config import SimulationConfig
    from rdspd_atheia.pipeline import run
    run(SimulationConfig())
"""

from __future__ import annotations

__version__ = "1.3.0"

from .config import SimulationConfig, smoke_config

__all__ = ["SimulationConfig", "smoke_config", "__version__"]
