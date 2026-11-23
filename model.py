import torch 
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden_dims = (), out_dim=10, use_bias=True, dropout=0, batch_norm=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #layers 
        layers = []
        prev_dim = in_dim
        for hidden_dim in (hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias, batch_norm=False))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim, bias=use_bias))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        logits = self.main(x)
        return F.log_softmax(logits, dim=1)
            
