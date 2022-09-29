from .entity_controller import EntityMAC
import torch as th


class VaeMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(VaeMAC, self).__init__(scheme, groups, args)
    
    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if kwargs.get('imagine', False):
            agent_outs, self.hidden_states, mean, std, z, zr, entities_repeat, entities_global, agent_mask, groups = self.agent(agent_inputs, self.hidden_states, test_mode=test_mode, **kwargs)
        else:
            agent_outs, self.hidden_states, mean, std, z, zr, entities_repeat, entities_global, agent_mask = self.agent(agent_inputs, self.hidden_states, test_mode=test_mode)

        if int_t:
            return agent_outs.squeeze(1), mean, std, z, zr, entities_repeat, entities_global, agent_mask
        if kwargs.get('imagine', False):
            return agent_outs, mean, std, z, zr, entities_repeat, entities_global, agent_mask, groups
        return agent_outs, mean, std, z, zr, entities_repeat, entities_global, agent_mask
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ret_agent_outs=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _, _, _, _, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs]
        return chosen_actions