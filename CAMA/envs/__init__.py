from functools import partial

from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env, StarCraft2CustomEnv
from .multiagent_particle_env import MultiAgentParticleEnv
from .starcraft2 import custom_scenario_registry as sc_scenarios
from .gridworld import CatchApple
from .traffic_junction import Entity_Traffic_Junction_Env


# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)#TODO
REGISTRY["sc2custom"] = partial(env_fn, env=StarCraft2CustomEnv)
REGISTRY["particle"] = partial(env_fn, env=MultiAgentParticleEnv)
REGISTRY["catch"] = partial(env_fn, env=CatchApple)
REGISTRY["traffic_junction"] = partial(env_fn, env=Entity_Traffic_Junction_Env)

s_REGISTRY = {}
s_REGISTRY.update(sc_scenarios)
