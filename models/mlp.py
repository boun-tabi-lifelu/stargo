import torch
import torch.nn as nn
import copy
from config import MLPModelConfig
from einops.layers.torch import Reduce
from einops import repeat
from transformers.activations import ACT2FN

class MLP(nn.Module):
    def __init__(self, config: MLPModelConfig):
        super().__init__()

        self.config = config

        self.mean_pool = Reduce("batch seq_len seq_emb_dim -> batch 1 seq_emb_dim", "mean")



        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(config.seq_input_dim + config.go_input_dim, config.hidden_layers[0]))
        self.layers.append(ACT2FN[config.activation])
        self.layers.append(nn.Dropout(config.dropout))
        for i in range(len(config.hidden_layers) - 1):
            self.layers.append(nn.Linear(config.hidden_layers[i], config.hidden_layers[i+1]))
            self.layers.append(ACT2FN[config.activation])
            self.layers.append(nn.Dropout(config.dropout))


        self.layers.append(nn.Linear(config.hidden_layers[-1], 1))
        # we use bce with logits in the loss function


    def forward(self, go_emb: torch.Tensor, seq_emb: torch.Tensor, **_):
        # go_emb: [batch, go_subontology_size, go_emb_dim]
        # seq_emb: [seq_len, seq_emb_dim]
        # need to transform to [go_subontology_size, go_emb_dim + seq_emb_dim]

        # mean pool the sequence embedding
        prot_emb = self.mean_pool(seq_emb)

        # broadcast to [go_subontology_size, seq_emb_dim]
        prot_emb = repeat(prot_emb, "batch 1 seq_emb_dim -> batch go_subontology_size seq_emb_dim", go_subontology_size=go_emb.shape[1])

        # concatenate the go embedding and the mean pooled sequence embedding
        x = torch.cat([go_emb, prot_emb], dim=-1)

        for layer in self.layers:
            x = layer(x)

        return x # [batch, go_subontology_size, 1]
