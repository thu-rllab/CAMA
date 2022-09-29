REGISTRY = {}


from .basic_controller import BasicMAC
from .entity_controller import EntityMAC
from .copa_controller import COPAMAC
from .icm_controller import ICMMAC
from .gat_controller import GatMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["entity_mac"] = EntityMAC
REGISTRY["copa_mac"] = COPAMAC
REGISTRY["icm_mac"] = ICMMAC
REGISTRY["gat_mac"] = GatMAC



