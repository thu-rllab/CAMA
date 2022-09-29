REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .entity_rnn_agent import ImagineEntityAttentionRNNAgent, EntityAttentionRNNAgent
from .entity_ff_agent import EntityAttentionFFAgent, ImagineEntityAttentionFFAgent
from .entity_copa_agent import EntityAttentionCOPAAgent, ImagineEntityAttentionCOPAAgent
from .entity_maic_agent import EntityAttentionMAICAgent, ImagineEntityAttentionMAICAgent
from .entity_imac_agent import EntityAttentionIMACAgent, ImagineEntityAttentionIMACAgent
from .entity_rnn_icm_agent import EntityAttentionRNNICMAgent, ImagineEntityAttentionRNNICMAgent
from .rnn_icm_agent import RNNICMAgent
from .entity_rnn_gat_agent import EntityAttentionRNNGATAgent, ImagineEntityAttentionRNNGATAgent


REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["entity_attend_ff"] = EntityAttentionFFAgent
REGISTRY["imagine_entity_attend_ff"] = ImagineEntityAttentionFFAgent
REGISTRY["entity_attend_rnn"] = EntityAttentionRNNAgent
REGISTRY["imagine_entity_attend_rnn"] = ImagineEntityAttentionRNNAgent
REGISTRY["entity_attend_copa"] = EntityAttentionCOPAAgent
REGISTRY["imagine_entity_attend_copa"] = ImagineEntityAttentionCOPAAgent
REGISTRY["entity_attend_maic"] = EntityAttentionMAICAgent
REGISTRY["imagine_entity_attend_maic"] = ImagineEntityAttentionMAICAgent
REGISTRY["entity_attend_imac"] = EntityAttentionIMACAgent
REGISTRY["imagine_entity_attend_imac"] = ImagineEntityAttentionIMACAgent
REGISTRY["entity_attend_rnn_icm"] = EntityAttentionRNNICMAgent
REGISTRY["imagine_entity_attend_rnn_icm"] = ImagineEntityAttentionRNNICMAgent
REGISTRY["rnn_icm"] = RNNICMAgent



REGISTRY["imagine_entity_attend_rnn_gat"] = ImagineEntityAttentionRNNGATAgent
REGISTRY["entity_attend_rnn_gat"] = EntityAttentionRNNGATAgent

