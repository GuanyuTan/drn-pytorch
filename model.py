import torch
import torch.nn as nn
from drn import DRNLayer
from rdrn import RDRNLayer


class DRN(nn.Module):
    def __init__(self, cfg):
        super(DRN, self).__init__()
        self.n_in = cfg.MODEL.N_IN
        self.n_out = cfg.MODEL.N_OUT
        self.n_nodes = cfg.MODEL.N_NODES
        self.hidden_q = cfg.MODEL.HIDDEN_Q
        self.q = cfg.MODEL.Q
        self.n_layers = cfg.MODEL.N_LAYERS
        self.layers = nn.ModuleList([])
        self.w_init_method = 'uniform'
        if hasattr(cfg.MODEL, 'INIT_METHOD'):
            self.w_init_method = cfg.MODEL.INIT_METHOD
        if hasattr(cfg.MODEL, 'INIT'):
            self.w_init = cfg.MODEL.INIT
        else:
            self.w_init = 0.1

        if self.n_layers == 0:
            self.layers.add_module(name="DRNLayer_out",
                                   module=DRNLayer(
                                       self.n_in,
                                       self.n_out,
                                       self.q,
                                       self.q,
                                       w_init=self.w_init,
                                       w_init_method=self.w_init_method,
                                   ))
        else:
            self.layers.add_module(name="DRNLayer_in",
                                   module=DRNLayer(
                                       n_lower=self.n_in,
                                       n_upper=self.n_nodes,
                                       q_lower=self.q,
                                       q_upper=self.hidden_q,
                                       w_init=self.w_init,
                                       w_init_method=self.w_init_method,
                                   ))
            for l in range(2, self.n_layers+1):
                self.layers.add_module(name=f'DRNLayer_{l}',
                                       module=DRNLayer(
                                           n_lower=self.n_nodes,
                                           n_upper=self.n_nodes,
                                           q_lower=self.hidden_q,
                                           q_upper=self.hidden_q,
                                           w_init=self.w_init,
                                           w_init_method=self.w_init_method,
                                       ))
            self.layers.add_module(name='DRNLayer_out',
                                   module=DRNLayer(
                                       n_lower=self.n_nodes,
                                       n_upper=self.n_out,
                                       q_lower=self.hidden_q,
                                       q_upper=self.q,
                                       w_init=self.w_init,
                                       w_init_method=self.w_init_method,
                                   ))

    def forward(self, P):
        for index, l in enumerate(self.layers):
            P = l(P)
        return P


class RDRN(nn.Module):
    def __init__(self, cfg):
        super(RDRN, self).__init__()
        self.n_in = cfg.MODEL.N_IN
        self.n_out = cfg.MODEL.N_OUT
        self.q_hidden = cfg.MODEL.HIDDEN_Q
        self.n_hidden = cfg.MODEL.N_HIDDEN
        self.q = cfg.MODEL.Q
        self.layers = nn.ModuleList([])
        self.layers.add_module(
            name='RDRNLayer',
            module=RDRNLayer(
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_in=self.n_in,
                q_in=self.q,
                q_out=self.q,
                q_hidden=self.q_hidden,
            )
        )

    def forward(self, P):
        for l in self.layers:
            P = l(P)
        return P


class MLP_bins(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg.MODEL.N_HIDDEN
        self.hidden_dim = cfg.MODEL.DIM_HIDDEN
        self.n_in = cfg.MODEL.IN
        self.n_out = cfg.MODEL.OUT
        self.layers = nn.ModuleList([
            nn.Linear(self.n_in, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.n_out),
        ])
        self.activation = nn.Softmax(dim=-1)
        for l in self.layers:
            torch.nn.init.xavier_normal_(l.weight)

    def forward(self, P):
        for i, l in enumerate(self.layers):
            P = l(P)
        return self.activation(P)


class MLP_basis(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.n_layers = cfg.MODEL.N_HIDDEN
        self.hidden_dim = cfg.MODEL.DIM_HIDDEN
        self.n_in = cfg.MODEL.IN
        self.n_out = cfg.MODEL.OUT
        self.input = nn.Linear(self.n_in, self.hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers)])
        self.output = nn.Linear(self.hidden_dim, self.n_out)

    def forward(self, P):
        P = P.squeeze()
        P = self.input(P)
        for i, l in enumerate(self.layers):
            P = l(P)
        P = self.output(P)
        return P
