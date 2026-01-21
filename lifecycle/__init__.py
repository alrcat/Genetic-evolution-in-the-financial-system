"""
生命周期模块 - 管理Agent的生死与繁殖

包含:
- Death: 死亡机制
- Reproduction: 繁殖机制
- Birth: 出生机制
"""

from lifecycle.death import DeathMechanism, DeathCertificate
from lifecycle.reproduction import ReproductionEngine, CrossoverType
from lifecycle.birth import BirthFactory

__all__ = [
    "DeathMechanism",
    "DeathCertificate",
    "ReproductionEngine",
    "CrossoverType",
    "BirthFactory",
]
