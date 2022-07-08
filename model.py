import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
import numpy as np


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]

        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, config):
        super(AVWDCRNN, self).__init__()
        num_nodes = config['capacity'] // config['groups']
        if config['capacity'] % config['groups'] != 0:
            num_nodes += 1
        if config['group_id'] == config['groups'] - 1:
            num_nodes = config['capacity'] - num_nodes * (config['groups'] - 1)
        self.num_nodes = num_nodes
        self.feature_dim = config['in_var']
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, config):
        super(AGCRN, self).__init__()
        num_nodes = config['capacity'] // config['groups']
        if config['capacity'] % config['groups'] != 0:
            num_nodes += 1
        if config['group_id'] == config['groups'] - 1:
            num_nodes = config['capacity'] - num_nodes * (config['groups'] - 1)
        self.num_nodes = num_nodes
        self.feature_dim = config['in_var']
        self.input_window = config['input_len']
        self.output_window = config['output_len']
        self.output_dim = config['out_var']
        self.hidden_dim = config['rnn_units']
        self.embed_dim = config['embed_dim']

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(config)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        if config['use_gpu']:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self._logger = getLogger()
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = batch
        init_state = self.encoder.init_hidden(source.shape[0])

        output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv(output)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        return output

    def masked_mae_torch(self, preds, labels, null_val=np.nan):
        labels[torch.abs(labels) < 1e-4] = 0
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels.ne(null_val)
        mask = mask.float()
        mask /= torch.mean(mask)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(torch.sub(preds, labels))
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)