import torch.nn as nn
from drn import DRNLayer


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

        if self.n_layers == 0:
            self.layers.add_module(name="DRNLayer_out",
                                   module=DRNLayer(self.n_in,
                                                   self.n_out,
                                                   self.q,
                                                   self.q))
        else:
            self.layers.add_module(name="DRNLayer_in",
                                   module=DRNLayer(
                                       n_lower=self.n_in,
                                       n_upper=self.n_nodes,
                                       q_lower=self.q,
                                       q_upper=self.hidden_q
                                   ))
            for l in range(2, self.n_layers+1):
                self.layers.add_module(name=f'DRNLayer_{l}',
                                       module=DRNLayer(
                                           n_lower=self.n_nodes,
                                           n_upper=self.n_nodes,
                                           q_lower=self.hidden_q,
                                           q_upper=self.hidden_q
                                       ))
            self.layers.add_module(name='DRNLayer_out',
                                   module=DRNLayer(n_lower=self.n_nodes,
                                                   n_upper=self.n_out,
                                                   q_lower=self.hidden_q,
                                                   q_upper=self.q))

    def forward(self,P):
        for l in self.layers:
            P = l(P)
        return P