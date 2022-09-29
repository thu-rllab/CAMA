import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer
from torch.distributions import Categorical
import numpy as np

class FiniteDist(nn.Module):
    def __init__(self, in_dim, out_dim, device, limit):
        super(FiniteDist, self).__init__()
        assert out_dim % 2 == 0
        self.limit=limit
        self.msg_dim = out_dim // 2
        self.fc1 = nn.Linear(in_dim, self.msg_dim)
        self.fc2 = nn.Linear(in_dim, self.msg_dim)
        self.max_logvar = nn.Parameter((th.ones((1, self.msg_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-th.ones((1, self.msg_dim)).float() * 10).to(device), requires_grad=False)
            
    def forward(self, x, limit=True):
        mean = self.fc1(x)
        if self.limit:
            mean=th.tanh(mean)
        logvar = self.fc2(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return th.cat([mean, logvar], dim=-1)


class EntityAttentionRNNICMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionRNNICMAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)
        else:
            self.attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim*(1+args.double_attn),
                                           args.pooling_type,
                                           args)
        assert (not args.sp_use_same_attn and args.reserve_ori_f) == False 
        if not args.sp_use_same_attn:
            self.sp_fc1 = nn.Linear(input_shape, args.attn_embed_dim)
            if args.pooling_type is None:
                self.sp_attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim*(1+args.double_attn), args)
            else:
                self.sp_attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.pooling_type,
                                           args)
        if args.reserve_ori_f:
            self.fc_i = nn.Linear(input_shape, args.attn_embed_dim)
            self.attn_i = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim, args)
        fc2_dim = args.attn_embed_dim*(1+args.self_loc+args.reserve_ori_f+args.double_attn)
        if args.mi_message and not args.add_q:
            fc2_dim += args.msg_dim
        self.fc2 = nn.Linear(fc2_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        if args.self_loc:
            self.self_fc = nn.Linear(input_shape, args.attn_embed_dim)
            if not args.sp_use_same_attn:
                self.sp_self_fc = nn.Linear(input_shape, args.attn_embed_dim)
        self.fc_a1 = nn.Linear(args.attn_embed_dim*(1+args.self_loc)*2, args.rnn_hidden_dim)
        self.fc_a2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.attn_weights =None
        if args.mi_message: #Having coach
            # self.fc1_coach = nn.Linear(input_shape, args.attn_embed_dim)
            # if args.pooling_type is None:
            #     self.coach_attn = EntityAttentionLayer(args.attn_embed_dim,
            #                                  args.attn_embed_dim,
            #                                  args.attn_embed_dim, args)
            # else:
            #     self.coach_attn = EntityPoolingLayer(args.attn_embed_dim,
            #                                args.attn_embed_dim,
            #                                args.attn_embed_dim,
            #                                args.pooling_type,
            #                                args)
            attn_out_dim = args.attn_embed_dim*(1+args.self_loc+args.reserve_ori_f+args.double_attn)

            if args.rnn_message:
                self.rnn_mess_b = nn.Linear(attn_out_dim*(1+self.args.club_mi), args.rnn_hidden_dim)
                self.rnn_message = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
                self.fc_msg = FiniteDist(args.rnn_hidden_dim, args.msg_dim*2, args.device, args.limit_msg)

            else:
                self.fc_msg = FiniteDist(attn_out_dim*(1+self.args.club_mi), args.msg_dim*2, args.device, args.limit_msg)
            
            self.fc_q = nn.Sequential(nn.Linear(attn_out_dim+args.club_mi*args.attn_embed_dim, args.attn_embed_dim),
            # self.fc_q = nn.Sequential(nn.Linear(attn_out_dim, args.attn_embed_dim),
                                        nn.ReLU(),
                                        FiniteDist(args.attn_embed_dim, args.msg_dim*2, args.device, args.limit_msg))
            if args.add_q:
                self.adhoc_q_net = nn.Sequential(nn.Linear(args.msg_dim, args.attn_embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(args.attn_embed_dim, args.n_actions))
        if args.private_entity_shape > 0: #Warning, this code only works for n_agent==n_entities
            self.fc_private = nn.Linear(args.attn_embed_dim+args.private_entity_shape, args.attn_embed_dim)
        self.attn_weights=None

    def init_hidden(self):
        # make hidden states on same device as model
        self.attn_weights=None
        if self.args.rnn_message:
            return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_(), self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()    
        self.msg = np.zeros([1,0,self.args.n_agents, self.args.msg_dim])
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def ICM(self, inputs_s, inputs_sp, hidden_state, imagine=False):
        if self.args.mi_message:
            q, hs, xs, _, _, _ = self.forward(inputs_s, hidden_state, return_F=True, only_F=False)
        else:
            q, hs, xs = self.forward(inputs_s, hidden_state, return_F=True, only_F=False)
        xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
        x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
        logits = self.fc_a2(x)
        bs, ts, _, _ = inputs_s[0].shape
        logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
        return q, hs, logits

    def MI_ICM(self, inputs_s, inputs_sp, hidden_state, randidx=None):
        assert self.args.mi_message == True
        q, hs, xs, zt, zt_logits, msg_q_logits = self.forward(inputs_s, hidden_state, return_F=True, only_F=False, randidx=randidx)
        bs, ts, _, _ = inputs_s[0].shape
        xs = xs.reshape(bs,ts, self.args.n_agents, self.args.attn_embed_dim)
        xsp = th.cat([xs[:,1:], xs[:,-1:]], dim=1)
        # xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
        x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
        logits = self.fc_a2(x)
        logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
        return q, hs, logits, zt, zt_logits, msg_q_logits

    def get_club_message(self, f1, f2, bs, ts, randidx=None, hidden_state=None):
        hs=None
        if self.args.rnn_message:
            x3_coach = F.relu(self.rnn_mess_b(th.cat([f1,f2], dim=-1)))
            x3_coach = x3_coach.reshape(bs, ts, self.args.n_agents, -1)
            h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hs = []
            for t in range(ts):
                curr_x3 = x3_coach[:, t].reshape(-1, self.args.rnn_hidden_dim)
                h = self.rnn_message(curr_x3, h)
                hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
            hs = th.stack(hs, dim=1)  # Concat over time
            zt_logits = self.fc_msg(hs)
        else:
            zt_logits = self.fc_msg(th.cat([f1,f2], dim=-1))
        zt_logits = zt_logits.reshape(bs,ts,self.args.n_agents, self.args.msg_dim*2)
        mean=zt_logits[:,:,:,:self.args.msg_dim]
        if self.args.save_entities_and_msg:
            self.msg = np.concatenate([self.msg, mean.detach().cpu().numpy()], axis=1)
        logvar = zt_logits[:,:,:,self.args.msg_dim:]
        zt_dis = th.distributions.Normal(mean, logvar.exp().sqrt())
        if self.training:
            zt = zt_dis.rsample()
        else:
            zt = zt_dis.mean
        if randidx is not None: 
            flattern_f1 = f1.reshape(bs, ts, self.args.n_agents, -1)
            flattern_f1 = flattern_f1[:,:-1].reshape(bs*(ts-1)*self.args.n_agents, -1) #del the last t corresponding to the mask.
            flattern_f2 = f2.reshape(bs, ts, self.args.n_agents, -1)
            flattern_f2 = flattern_f2[:,:-1].reshape(bs*(ts-1)*self.args.n_agents, -1)
            oriidx = randidx.chunk(2, dim=0)[0]
            oriidx = oriidx.repeat(2)
            if self.args.rnn_message: #need a q to estimate p.
                msg_q_logits_club = self.fc_q(th.cat([flattern_f1[randidx], flattern_f2[oriidx]], dim=-1)) #for club
                msg_q_logits_logq = self.fc_q(th.cat([f1.detach(), f2.detach()], dim=-1))
                msg_q_logits = [msg_q_logits_club, msg_q_logits_logq]
            else:
                msg_q_logits = self.fc_msg(th.cat([flattern_f1[randidx], flattern_f2[oriidx]], dim=-1))
        else:
            msg_q_logits = zt_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim*2)
        return zt, zt_logits, msg_q_logits, hs
        

    def get_coach_message(self, inputs, feature, randidx=None, hidden_state=None):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                                   (1 - entity_mask.to(th.float)).unsqueeze(1))
        x1_coach = F.relu(self.fc1_coach(entities))
        x2_coach = self.coach_attn(x1_coach, pre_mask=attn_mask.to(th.uint8),
                       post_mask=agent_mask)
        hs=None
        if self.args.rnn_message:
            x3_coach = F.relu(self.rnn_mess_b(x2_coach))
            x3_coach = x3_coach.reshape(bs, ts, self.args.n_agents, -1)
            h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hs = []
            for t in range(ts):
                curr_x3 = x3_coach[:, t].reshape(-1, self.args.rnn_hidden_dim)
                h = self.rnn_message(curr_x3, h)
                hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
            hs = th.stack(hs, dim=1)  # Concat over time
            zt_logits = self.fc_msg(hs)
        else:
            zt_logits = self.fc_msg(x2_coach)
        zt_logits = zt_logits.reshape(bs,ts,self.args.n_agents, self.args.msg_dim*2)
        mean=zt_logits[:,:,:,:self.args.msg_dim]
        logstd = self.max_logvar - F.softplus(self.max_logvar - zt_logits[:,:,:,self.args.msg_dim:])
        logstd = self.min_logvar + F.softplus(logstd - self.min_logvar)
        zt_dis = th.distributions.Normal(mean, logstd.exp().sqrt())
        if self.training:
            if randidx is not None:
                zt = zt_dis.rsample()
            else:
                zt = zt_dis.sample()
        else:
            zt = zt_dis.mean
        if self.args.club_mi:
            if randidx is not None: 
                flattern_feature = feature.reshape(bs*ts*self.args.n_agents, -1)
                flattern_x2_coach = x2_coach.reshape(bs*ts*self.args.n_agents, -1)
                oriidx = randidx.chunk(2, dim=0)[0]
                oriidx = oriidx.repeat(2)
                # msg_q_logits = self.fc_q(flattern_feature[randidx])
                msg_q_logits = self.fc_q(th.cat([flattern_feature[randidx], flattern_x2_coach[oriidx]], dim=-1))
            else:
                msg_q_logits = self.fc_q(th.cat([feature, x2_coach], dim=-1))
                # msg_q_logits = self.fc_q(feature)
                msg_q_logits = msg_q_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim*2)
        else:
            msg_q_logits = self.fc_q(feature)
            msg_q_logits = msg_q_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim*2)
        return zt, zt_logits, msg_q_logits, hs
        

    def get_feature(self, inputs, return_F=False, only_F=False, rank_percent=None, pre_obs_mask=False, ret_attn_weights=False):
        entities, obs_mask, entity_mask = inputs
        if self.args.private_entity_shape > 0:
            private_entities = entities[:, :, :, :self.args.private_entity_shape]
            entities = entities[:, :, :, self.args.private_entity_shape:]
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        if pre_obs_mask:
            obs_mask = obs_mask.reshape(bs * ts * self.args.attn_n_heads, self.args.n_agents, ne)
        else:
            obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        # if self.args.reserve_ori_f:
        #     x1_i = F.relu(self.fc_i(entities))
        #     x2_i = self.attn_i(x1_i, pre_mask=obs_mask, post_mask=agent_mask)
        #     if return_F and only_F:
        #         return x2_i
        if only_F and not self.args.sp_use_same_attn: #means it is s_p and needs other nets
            cur_fc1 = self.sp_fc1
            cur_attn = self.sp_attn
        else:
            cur_fc1 = self.fc1
            cur_attn = self.attn
        x1 = F.relu(cur_fc1(entities))
        if rank_percent is not None:
            x2, true_pre_mask = cur_attn(x1, pre_mask=obs_mask, post_mask=agent_mask, rank_percent=rank_percent, entity_mask=entity_mask, ret_attn_weights=ret_attn_weights)
        else:
            x2 = cur_attn(x1, pre_mask=obs_mask, post_mask=agent_mask, ret_attn_weights=ret_attn_weights)
        if ret_attn_weights:
            x2, attn_weights = x2
            if self.attn_weights is None:
                self.attn_weights = attn_weights
            else:
                self.attn_weights=th.cat([self.attn_weights, attn_weights], dim=0)

        # for i in range(self.args.repeat_attn):
        #     x2 = cur_attn(x2, pre_mask=obs_mask, post_mask=agent_mask)
        if self.args.private_entity_shape > 0:
            private_entities = private_entities.reshape(bs*ts, ne, self.args.private_entity_shape)
            x2 = F.relu(self.fc_private(th.cat([private_entities, x2], dim=2)))
        if return_F and only_F:
            return x2
        if rank_percent is not None:
            return x2, agent_mask, bs, ts, true_pre_mask.reshape(bs, ts, self.args.attn_n_heads,self.args.n_agents,ne)
        return x2, agent_mask, bs, ts
    def get_q(self, x2, agent_mask, bs, ts, hidden_state, return_F=False, zt=None, h1=None):
        if zt is not None and not self.args.add_q:
            _, _, na, edim = zt.shape
            if self.args.no_msg:
                x3 = F.relu(self.fc2(th.cat([x2, th.zeros(bs*ts, na, edim).to(x2.device)], dim=-1)))
            else:
                x3 = F.relu(self.fc2(th.cat([x2, zt.reshape(bs*ts, na, edim)], dim=-1)))
        else: 
            x3 = F.relu(self.fc2(x2))
        x3 = x3.reshape(bs, ts, self.args.n_agents, -1)
        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x3 = x3[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x3, h)
            hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time

        q = self.fc3(hs)
        # zero out output for inactive agents
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)
        # q = q.reshape(bs * self.args.n_agents, -1)
        if zt is not None and self.args.add_q:
            self.adhoc_q = self.adhoc_q_net(zt)
            if not self.args.no_msg:
                q += self.adhoc_q
        if h1 is not None:
            hs = [h1, hs]
        if return_F:
            return q, hs, x2
        return q, hs
    
    def get_inputs_m(self, inputs, true_pre_mask=None):
        entities, obs_mask, entity_mask = inputs
        if true_pre_mask is not None:
            c_mask = self.logical_or(self.logical_not(true_pre_mask), \
                self.entitymask2attnmask(entity_mask)[:,:,:self.args.n_agents,:].unsqueeze(2)) #bs, ts, n_head, na, ne
            inputs_m = (entities, c_mask, entity_mask)
        else:    
            c_mask = self.logical_or(self.logical_not(obs_mask), self.entitymask2attnmask(entity_mask))
            entities = entities.repeat(2, 1, 1, 1)
            obs_mask = th.cat([obs_mask, c_mask], dim=0)
            entity_mask = entity_mask.repeat(2, 1, 1)
            inputs_m = (entities, obs_mask, entity_mask)
        return inputs_m


    def forward(self, inputs, hidden_state, return_F=False, only_F=False, randidx=None, ret_attn_weights=False):
        if return_F and only_F:
            return self.get_feature(inputs, return_F=return_F, only_F=only_F,ret_attn_weights=ret_attn_weights)
        if self.args.mi_message:
            if self.args.rnn_message:
                h1, h2 = hidden_state
            else:
                h1, h2 = None, hidden_state
            if self.args.club_mi:
                #msg_q_logits_i=f(s^g_i, s^l_j)
                x2, agent_mask, bs, ts, true_pre_mask = self.get_feature(inputs, return_F=return_F, only_F=only_F, rank_percent=self.args.rank_percent, ret_attn_weights=ret_attn_weights)
                inputs_m = self.get_inputs_m(inputs, true_pre_mask=true_pre_mask)
                x2_m, _, _, _ = self.get_feature(inputs_m, return_F=return_F, only_F=only_F, pre_obs_mask=True)
                zt, zt_logits, msg_q_logits, h1 = self.get_club_message(x2, x2_m, bs, ts, randidx=randidx, hidden_state=h1)
            else:
                inputs_m = self.get_inputs_m(inputs)
                x2, agent_mask, bs, ts = self.get_feature(inputs_m, return_F=return_F, only_F=only_F, ret_attn_weights=ret_attn_weights)
                x2, x2_m = x2.chunk(2, dim=0)
                agent_mask = agent_mask.chunk(2, dim=0)[0]
                bs //= 2
                zt, zt_logits, msg_q_logits, h1 = self.get_coach_message(inputs, x2_m, hidden_state=h1)
            return self.get_q(x2, agent_mask, bs, ts, h2, return_F=return_F, zt=zt, h1=h1)+(zt, zt_logits, msg_q_logits)
        else:
            x2, agent_mask, bs, ts = self.get_feature(inputs, return_F=return_F, only_F=only_F, ret_attn_weights=ret_attn_weights)
            return self.get_q(x2, agent_mask, bs, ts, hidden_state, return_F=return_F)
    
    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out
    def entitymask2attnmask(self, entity_mask):
        bs, ts, ne = entity_mask.shape
        # agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - entity_mask.to(th.float)).reshape(bs * ts, ne, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)

class ImagineEntityAttentionRNNICMAgent(EntityAttentionRNNICMAgent):
    def __init__(self, *args, **kwargs):
        super(ImagineEntityAttentionRNNICMAgent, self).__init__(*args, **kwargs)
        if self.args.group == "dpp":
            self.cos = nn.CosineSimilarity(dim=3)
            self.ally=None
    
    def ICM(self, inputs_s, inputs_sp, hidden_state, imagine=True):
        assert self.args.mi_message == False
        if imagine:
            q, hs, xs, m = self.forward(inputs_s, hidden_state, imagine=imagine, return_F=True, only_F=False)
            xs = xs.chunk(3, dim=0)[0]
        else:
            q, hs, xs = self.forward(inputs_s, hidden_state, imagine=imagine, return_F=True, only_F=False)
        xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
        x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
        logits = self.fc_a2(x)
        bs, ts, _, _ = inputs_s[0].shape
        logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
        if imagine:
            return q, hs, m, logits
        else:
            return q, hs, logits
    def MI_ICM(self, inputs_s, inputs_sp, hidden_state, imagine=True, randidx=None):
        assert self.args.mi_message == True
        if imagine:
            q, hs, xs, zt, zt_logits, msg_q_logits, m = self.forward(inputs_s, hidden_state, imagine=imagine, return_F=True, only_F=False, randidx=randidx)
            xs = xs.chunk(3, dim=0)[0]
        else:
            q, hs, xs, zt, zt_logits, msg_q_logits = self.forward(inputs_s, hidden_state, imagine=imagine, return_F=True, only_F=False, randidx=randidx)
        bs, ts, _, _ = inputs_s[0].shape
        xs = xs.reshape(bs,ts, self.args.n_agents, self.args.attn_embed_dim)
        xsp = th.cat([xs[:,1:], xs[:,-1:]], dim=1)
        # xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
        x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
        logits = self.fc_a2(x)
        logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
        if imagine:
            return q, hs, m, logits, zt, zt_logits, msg_q_logits
        else:
            return q, hs, logits, zt, zt_logits, msg_q_logits

    # def get_dpp_mask(self, inputs):
    #     with th.no_grad():
    #         x2, _, agent_mask, _, _ = self.get_feature(inputs)#[bs*ts, na, ed] 
    #         entities, obs_mask, entity_mask = inputs
    #         bs, ts, ne, ed = entities.shape
    #         cos_matrix = self.cos(x2.unsqueeze(1),x2.unsqueeze(2)) #[bs*ts, na, na]
    #         mask  = 2 ** th.arange(self.args.n_agents - 1, -1, -1).to(cos_matrix.device)
    #         x = th.arange(2**self.args.n_agents).to(cos_matrix.device)
    #         y = x.unsqueeze(-1).bitwise_and(mask).ne(0) #[2^n,n] from [0,0,0,...,0] to [1,1,1,...,1]
    #         m = th.logical_and(y.unsqueeze(1), y.unsqueeze(-1)) #[2^n, n, n]
    #         #get the determinant of all subsets
    #         cofactor_matrix = cos_matrix.unsqueeze(1) * m.unsqueeze(0) #[bs*ts, 2^n, n, n]
    #         missed_ones = th.diag(th.ones(self.args.n_agents)).unsqueeze(0).to(cos_matrix.device)*th.logical_not(m) #[2^n,n,n]
    #         cofactor_matrix += missed_ones.unsqueeze(0)
    #         p = th.linalg.det(cofactor_matrix) #[bs*ts, 2^n]
    #         p[p<0] = 0 #Fix the numerical error
    #         need_fill = th.logical_and(y.unsqueeze(0), agent_mask.unsqueeze(1)).sum(2).ne(0) #[bs*ts, 2^n], True means discard the p
    #         p = p.masked_fill(th.logical_or(th.logical_or(need_fill, th.isnan(p)), th.isinf(p)), 0)
    #         invalid_mask = p.sum(1)==0
    #         p[:,0] = p[:,0].masked_fill(invalid_mask, 1.0)
    #         p = p.reshape(bs, ts, 2**self.args.n_agents)
    #         p = p.sum(1) #[bs, 2^n]
    #         try:
    #             cat = Categorical(probs=p)
    #         except:
    #             error_dict = {'p':p.detach().cpu().numpy(), 'agent_mask':agent_mask.detach().cpu().numpy(), 'z':x2.detach().cpu().numpy()}
    #             import pickle
    #             with open("./error_dict.pkl", "wb") as f:
    #                 pickle.dump(error_dict, f)
    #             print('Saving p, exit.')
    #             exit(0)
    #         cat = Categorical(probs=p)
    #         s = cat.sample()
    #         sample_mask = th.logical_not(s.unsqueeze(-1).bitwise_and(mask).ne(0)) #[bs,na]
    #     return sample_mask



    def forward(self, inputs, hidden_state, imagine=False, return_F=False, only_F=False, randidx=None, ret_attn_weights=False):
        if not imagine:
            return super(ImagineEntityAttentionRNNICMAgent, self).forward(inputs, hidden_state, return_F=return_F, only_F=only_F, randidx=randidx, ret_attn_weights=ret_attn_weights)
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        if self.args.group == 'random':
            groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)
            groupA = th.bernoulli(groupA_probs).to(th.uint8)
        elif self.args.group == 'dpp':
            self.ally = self.get_dpp_mask(inputs) #[bs,na]
            enemy_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne-self.args.n_agents)
            enemy = th.bernoulli(enemy_probs).to(th.uint8)
            groupA = th.cat([self.ally.unsqueeze(1), enemy], dim=2)
        else:
            raise NotImplementedError
        
        groupB = self.logical_not(groupA)
        # mask out entities not present in env
        groupA = self.logical_or(groupA, entity_mask[:, [0]])
        groupB = self.logical_or(groupB, entity_mask[:, [0]])

        # convert entity mask to attention mask
        groupAattnmask = self.entitymask2attnmask(groupA)
        groupBattnmask = self.entitymask2attnmask(groupB)
        # create attention mask for interactions between groups
        interactattnmask = self.logical_or(self.logical_not(groupAattnmask),
                                           self.logical_not(groupBattnmask))
        # get within group attention mask
        withinattnmask = self.logical_not(interactattnmask)

        activeattnmask = self.entitymask2attnmask(entity_mask[:, [0]])
        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self.logical_or(withinattnmask, activeattnmask)
        Iattnmask_noobs = self.logical_or(interactattnmask, activeattnmask)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self.logical_or(withinattnmask, obs_mask)
        interactattnmask = self.logical_or(interactattnmask, obs_mask)

        entities = entities.repeat(3, 1, 1, 1)
        obs_mask = th.cat([obs_mask, withinattnmask, interactattnmask], dim=0)
        entity_mask = entity_mask.repeat(3, 1, 1)

        inputs = (entities, obs_mask, entity_mask)
        if self.args.rnn_message:
            hidden_state = [h.repeat(3, 1, 1) for h in hidden_state]
        else:
            hidden_state = hidden_state.repeat(3, 1, 1)
        if self.args.mi_message:
            q, h, f, zt, zt_logits, msg_q_logits= super(ImagineEntityAttentionRNNICMAgent, self).forward(inputs, hidden_state, return_F=return_F, only_F=only_F, randidx=randidx, ret_attn_weights=ret_attn_weights)
            return q, h, f, zt, zt_logits, msg_q_logits, (Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1))
        else:
            q, h, f = super(ImagineEntityAttentionRNNICMAgent, self).forward(inputs, hidden_state, return_F=return_F, only_F=only_F, ret_attn_weights=ret_attn_weights)
            return q, h, f, (Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1))
