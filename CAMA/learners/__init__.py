from telnetlib import GA
from .q_learner import QLearner
from .copa_q_learner import COPAQLearner
from .vae_q_learner import VaeQLearner
from .icm_q_learner import ICMQLearner
from .gat_q_learner import GatQLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["copa_q_learner"] = COPAQLearner
REGISTRY["vae_q_learner"] = VaeQLearner
REGISTRY["icm_q_learner"] = ICMQLearner
REGISTRY["gat_q_learner"] = GatQLearner

