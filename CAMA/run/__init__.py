from .run import run as default_run
from .icm_run import run as icm_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["icm"] = icm_run
