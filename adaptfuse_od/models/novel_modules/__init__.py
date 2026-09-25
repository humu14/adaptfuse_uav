# AdapFuse-OD: Novel Modules
from .spd_conv import SPDConv, SPDBlock
from .seam import SEAM
from .p2_head import P2Module, P2DetectionHead
from .nad_head import NADHead
from .acr import ACR
from .hazard_prior_gate import HazardPriorGate, HazardAwareStem, attach_hazard_prior_gate

__all__ = [
    "SPDConv", "SPDBlock", "SEAM", "P2Module", "P2DetectionHead", "NADHead", "ACR",
    "HazardPriorGate", "HazardAwareStem", "attach_hazard_prior_gate",
]
