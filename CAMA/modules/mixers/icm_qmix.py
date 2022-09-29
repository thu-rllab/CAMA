import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer


class AttentionICMHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(AttentionICMHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = args.entity_shape
        if self.args.entity_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(hypernet_embed,
                                             hypernet_embed,
                                             hypernet_embed, args)
        else:
            self.attn = EntityPoolingLayer(hypernet_embed,
                                           hypernet_embed,
                                           hypernet_embed,
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None, return_f=False):
        x1 = F.relu(self.fc1(entities))
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask) 
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2).bool(), 0) #[bs, na, edim]
        if self.mode == 'vector':
            x3 = x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            x3 = x3.mean(dim=2)
        elif self.mode == 'scalar':
            x3 = x3.mean(dim=(1, 2))
        if return_f:
            return x3, x2
        return x3


class ICMQMixer(nn.Module):
    def __init__(self, args):
        super(ICMQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionICMHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionICMHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionICMHyperNet(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = AttentionICMHyperNet(args, mode='scalar')

        self.non_lin = F.elu
        if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
            self.non_lin = F.tanh
        if args.global_icm:
            self.hyper_fc1 = nn.Linear(args.hypernet_embed*8, args.hypernet_embed*2)
            self.hyper_fc2 = nn.Linear(args.hypernet_embed*2, args.n_actions)
    
    def predict_action(self, s, sp):
        ac = self.hyper_fc1(th.cat([s,sp], dim=3))
        ac = self.hyper_fc2(ac) #[bs, t, n_agent, na]
        return ac

    def forward(self, agent_qs, inputs, imagine_groups=None, return_f=False):
        entities, entity_mask = inputs
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, 1, self.n_agents * 2) #[4800,1,16]
            Wmask, Imask = imagine_groups
            w1_W = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Wmask.reshape(bs * max_t,
                                                          ne, ne)) #[4800,8,32]
            w1_I = self.hyper_w_1(entities, entity_mask,
                                  attn_mask=Imask.reshape(bs * max_t,
                                                          ne, ne)) #[4800,8,32]
            w1 = th.cat([w1_W, w1_I], dim=1) #[4800,16,32]
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)#[4800,1,8]
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask, return_f = return_f) #[4800,8,32]
            if return_f:
                w1, x2_w1 = w1
        b1 = self.hyper_b_1(entities, entity_mask, return_f = return_f)#[4800,32]
        if return_f:
            b1, x2_b1=b1
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        if self.args.softmax_mixing_weights:
            w1 = F.softmax(w1, dim=-1)
        else:
            w1 = th.abs(w1)

        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1) #[4800,1,32]
        # Second layer
        w_final = self.hyper_w_final(entities, entity_mask, return_f=return_f)
        if return_f:
            w_final, x2_wf = w_final
        if self.args.softmax_mixing_weights:
            w_final = F.softmax(w_final, dim=-1) #[4800,32]
        else:
            w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1) 
        # State-dependent bias
        v = self.V(entities, entity_mask, return_f=return_f)
        if return_f:
            v, x2_v = v
        v = v.view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        if return_f:
            x2 = th.cat([x2_w1, x2_b1, x2_wf, x2_v], dim=2) #[bs,na, hypernet_embed*4]
            x2 = x2.reshape(bs, max_t, self.n_agents, self.args.hypernet_embed * 4)
            return q_tot, x2

        return q_tot
