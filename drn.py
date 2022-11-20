import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init

def cal_mult_bias(B, q):
    b_a, b_q, lamb_a, lamb_q = B
    # b_a : absolute bias
    # b_q : quadratic bias
    s0_np = np.arange(q, dtype=np.float32).reshape((1, q))
    s0 = torch.from_numpy(s0_np)
    expB = torch.exp(-b_q * torch.pow(s0/q - lamb_q, 2) - b_a * torch.abs(s0 / q - lamb_a))
    return expB

def init_D(q_upper, q_lower):
    D_np = np.zeros([q_upper, q_lower])
    for s1 in range(q_upper):
        for s0 in range(q_lower):
            D_np[s1, s0] = np.exp(-((float(s0)/q_lower - float(s1)/q_upper) ** 2))
    return D_np

def KL_div(y, y_hat):
    y = torch.clamp(y, 1e-15, 1.0)
    y_hat = torch.clamp(y_hat, 1e-15, 1.0)
    return torch.sum(y * torch.log(y/y_hat), dim=2)

def Ldjs(P,Q):
    M = (P+Q)/2.0
    l = 0.5*KL_div(P,M)+0.5*KL_div(Q,M)
    return torch.mean(l)/torch.log(torch.tensor(2.0))

def cal_logexp_bias(B, q):
    b_a, b_q, lamb_a, lamb_q = B
    # b_a : absolute bias
    # b_q : quadratic bias
    s0_np = np.arange(q, dtype=np.float32).reshape((1, q))
    s0 = torch.from_numpy(s0_np)
    B = -b_q * torch.pow(s0/q - lamb_q, 2) - b_a * torch.abs(s0 / q - lamb_a)
    return B

class DRNLayer(Module):
    def __init__(self, n_lower, n_upper, q_lower, q_upper, loadparam=False, loadW=None, loadB=None, fixWb=False, w_init=0.1, w_init_method='uniform'):
        super().__init__()
        self.n_lower, self.n_upper, self.q_lower, self.q_upper = n_lower, n_upper, q_lower, q_upper
        self.weight = Parameter(torch.empty((n_upper, n_lower)))
        self.bias_abs = Parameter(torch.empty((n_upper, 1))) 
        self.bias_q = Parameter(torch.empty((n_upper, 1))) 
        self.lambda_abs = Parameter(torch.empty((n_upper, 1))) 
        self.lambda_q = Parameter(torch.empty((n_upper, 1)))
        self.D = torch.from_numpy(init_D(q_upper, q_lower)).reshape([q_upper,q_lower,1,1])
        self.D = torch.tile(self.D, [1,1,n_upper, n_lower])
        self.reset_parmeters(w_init, w_init_method)
        
    def reset_parmeters(self, w_init, w_init_method):
        if w_init_method == 'uniform':
            init.uniform_(self.weight, -w_init, w_init )
            init.uniform_(self.bias_abs, -w_init, w_init )
            init.uniform_(self.bias_q, -w_init, w_init )
            init.uniform_(self.lambda_abs, 0.0, 1.0)
            init.uniform_(self.lambda_q, 0.0, 1.0 )
        elif w_init_method == 'normal':
            init.normal_(self.weight)
            init.normal_(self.bias_abs)
            init.normal_(self.bias_q)
            init.uniform_(self.lambda_abs)
            init.uniform_(self.lambda_q)
        elif w_init_method == 'xavier_normal':
            init.xavier_normal_(self.weight)
            init.xavier_normal_(self.bias_abs)
            init.xavier_normal_(self.bias_q)
            init.uniform_(self.lambda_abs)
            init.uniform_(self.lambda_q)

    def forward(self, P:torch.Tensor):
        P_tile = torch.tile(P.reshape([-1,1,self.n_lower, self.q_lower, 1]),[1, self.n_upper, 1, 1, 1])
        # print(self.D.shape)
        # print(self.weight.shape)
        T = torch.permute(torch.pow(self.D, self.weight), [2,3,0,1])
        # n_upper x n_lower x q_upper x q_lower
        Pw_unclipped = torch.squeeze(torch.einsum('jklm,ijkmn->ijkln', T, P_tile), dim=4)
        # batch_size x n_upper x n_lower x q_upper
        Pw = torch.clamp(Pw_unclipped, 1e-15, 1e+15)
        # underflow handling
        # 1. Log each term in Pw
        logPw = torch.log(Pw)
        # 2. sum over neighbors over n_lower axis (2), capital PI notation over integrated variables
        logsum = torch.sum(logPw, axis=[2])
        # 3. log of exp of bias terms
        exponent_B = cal_logexp_bias([self.bias_abs, self.bias_q, self.lambda_abs, self.lambda_q],self.q_upper)
        # 4. add B to logsum
        logsumB = torch.add(logsum, exponent_B)
        # 5. find max over s0
        max_logsum = torch.max(logsumB,dim=2, keepdim=True)
        # 6. subtract max_logsum and exponentiate (the max term will have a result of exp(0) = 1, preventing underflow)
        expm_P = torch.exp(torch.subtract(logsumB, max_logsum.values))
        # normalize
        Z = torch.sum(expm_P, 2, keepdim=True)
        ynorm = torch.divide(expm_P, Z)
        return ynorm

