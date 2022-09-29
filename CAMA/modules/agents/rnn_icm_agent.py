import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNICMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim*(1+self.args.reserve_ori_f), args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        if args.reserve_ori_f:
            self.fc_i = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc_icm1 = nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim)
        self.fc_icm2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


        

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def ICM(self, inputs_s, inputs_sp, hidden_state):
        q, hs, xs = self.forward(inputs_s, hidden_state, return_F=True, only_F=False)
        xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
        x = F.relu(self.fc_icm1(th.cat([xs, xsp], dim=-1)))
        logits = self.fc_icm2(x)
        bs, ts, _, _ = inputs_s.shape
        logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
        return q, hs, logits

    def get_feature(self, inputs, return_F=False, only_F=False): #TODO 
        bs, ts, na, _ = inputs.shape
        sp = (bs, ts, na)
        if self.args.reserve_ori_f:
            x1_i = F.relu(self.fc_i(inputs))
            if return_F and only_F:
                return x1_i
        x1 = F.relu(self.fc1(inputs))
        if self.args.reserve_ori_f:
            x1 = th.cat([x1, x1_i], dim=-1)
        else:
            x1_i = x1
        if return_F and only_F:
            return x1_i
        return x1, x1_i, sp
    def get_q(self, x1, x1_i,  hidden_state, sp, return_F=False):
        bs, ts, na = sp
        x1 = F.relu(self.fc2(x1))
        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x = x1[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x, h)
            hs.append(h.view(bs, na, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time
        q = self.fc3(hs)
        if return_F:
            return q, hs, x1_i
        return q, hs
    def forward(self, inputs, hidden_state, return_F=False, only_F=False):
        if return_F and only_F:
            return self.get_feature(inputs, return_F=return_F, only_F=only_F)
        x1, x1_i, sp = self.get_feature(inputs, return_F=return_F, only_F=only_F)
        return self.get_q(x1, x1_i, hidden_state, sp, return_F=return_F)
