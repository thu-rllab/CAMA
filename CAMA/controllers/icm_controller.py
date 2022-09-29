from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .entity_controller import EntityMAC

class ICMMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(ICMMAC, self).__init__(scheme, groups, args)
    
    def forward(self, ep_batch, t, test_mode=False, train_mode=False, imagine=False, randidx=None, ret_attn_weights=False):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True
        if self.args.mi_message:
            if train_mode:
                agent_inputs, agent_inputs_sp = self._build_inputs(ep_batch, t, sp=True)
                if imagine:
                    agent_outs, self.hidden_states, groups, logits, zt, zt_logits, msg_q_logits = self.agent.MI_ICM(agent_inputs, agent_inputs_sp, self.hidden_states, randidx=randidx)
                else:
                    agent_outs, self.hidden_states, logits, zt, zt_logits, msg_q_logits = self.agent.MI_ICM(agent_inputs, agent_inputs_sp, self.hidden_states, randidx=randidx)
            else:
                agent_inputs = self._build_inputs(ep_batch, t)
                agent_outs, self.hidden_states, _, _, _ = self.agent(agent_inputs, self.hidden_states)
            if int_t:
                return agent_outs.squeeze(1)
            if train_mode:
                if imagine:
                    return agent_outs, groups, logits, zt, zt_logits, msg_q_logits 
                else:
                    return agent_outs, logits, zt, zt_logits, msg_q_logits 
            return agent_outs
        else:
            if train_mode:
                agent_inputs, agent_inputs_sp = self._build_inputs(ep_batch, t, sp=True)
                if imagine:
                    agent_outs, self.hidden_states, groups, logits = self.agent.ICM(agent_inputs, agent_inputs_sp, self.hidden_states)
                else:
                    agent_outs, self.hidden_states, logits = self.agent.ICM(agent_inputs, agent_inputs_sp, self.hidden_states)
            else:
                agent_inputs = self._build_inputs(ep_batch, t)
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            if int_t:
                return agent_outs.squeeze(1)
            if train_mode:
                if imagine:
                    return agent_outs, groups, logits
                else:
                    return agent_outs, logits
            return agent_outs
    
    def init_hidden(self, batch_size):
        if self.args.rnn_message:
            self.hidden_states = self.agent.init_hidden()
            self.hidden_states = [x.unsqueeze(0).expand(batch_size, self.n_agents, -1) for x in self.hidden_states]
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def _build_inputs(self, batch, t, sp=False):
        # Assumes homogenous agents with entity + observation mask inputs.
        bs = batch.batch_size
        entities = []
        entities.append(batch["entities"][:, t])  # bs, ts, n_entities, vshape
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entities.append(ent_acs)
        entities = th.cat(entities, dim=3)
        if self.args.gt_mask_avail:
            agent_inputs = (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t], batch["gt_mask"][:, t])
        else:
            agent_inputs = (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t])
        if not sp:
            return agent_inputs
        # the last dim is not valid. 
        agent_inputs_sp = (th.cat([agent_inputs[0][:,1:], agent_inputs[0][:, -1:]], dim=1),
                           th.cat([agent_inputs[1][:,1:], agent_inputs[1][:, -1:]], dim=1),
                           th.cat([agent_inputs[2][:,1:], agent_inputs[2][:, -1:]], dim=1),)
        return agent_inputs, agent_inputs_sp

    def _get_input_shape(self, scheme):
        input_shape = scheme["entities"]["vshape"]
        if self.args.entity_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.private_entity_shape > 0:
            input_shape -= self.args.private_entity_shape
        return input_shape