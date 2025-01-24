import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder

sys.path.append(".")
from config import BertModelConfig


class BERTBasedModel(nn.Module):
    def __init__(self, config: BertModelConfig):
        super().__init__()

        self.config = config

        # Input projections
        self.seq_projection = nn.Linear(config.seq_input_dim, config.hidden_dim)

        if config.train_go_embeddings:
            self.go_projection = nn.Embedding(config.go_input_dim, config.hidden_dim)
        else:
            self.go_projection = nn.Linear(config.go_input_dim, config.hidden_dim)

        # Create BertConfig for encoder and decoder
        bert_config = BertConfig(
            hidden_size=config.hidden_dim,
            num_hidden_layers=config.num_encoder_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            _attn_implementation="sdpa"
        )

        # Encoder (sequence)
        self.encoder = BertEncoder(bert_config)

        # Decoder config with cross-attention
        decoder_config = BertConfig(
            hidden_size=config.hidden_dim,
            num_hidden_layers=config.num_decoder_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            is_decoder=True,
            add_cross_attention=True,
            _attn_implementation="sdpa"
        )

        # Decoder (GO terms)
        self.decoder = BertEncoder(decoder_config)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            # nn.Sigmoid()
            # sigmoid is performed inside binary_cross_entropy_with_logits in model.py
        )

    def forward(self, go_emb: torch.Tensor, seq_emb: torch.Tensor, attention_mask_go_emb: Optional[torch.Tensor] = None, attention_mask_seq_emb: Optional[torch.Tensor] = None):
        # Project inputs to hidden dimension
        seq_hidden = self.seq_projection(seq_emb)
        go_hidden = self.go_projection(go_emb)
        batch_size, terms_len = go_emb.shape[:2]

        if attention_mask_seq_emb is not None and attention_mask_seq_emb.ndim == 2:
            attention_mask_seq_emb = attention_mask_seq_emb[:, None, None, :]

            attention_mask_seq_emb = attention_mask_seq_emb.to(dtype=seq_emb.dtype)
            attention_mask_seq_emb = (1.0 - attention_mask_seq_emb) * torch.finfo(seq_emb.dtype).min

        if attention_mask_go_emb is None and self.config.decoder is False:
            attention_mask_go_emb = torch.ones(batch_size, terms_len, device=go_emb.device)

        if attention_mask_go_emb is not None and attention_mask_go_emb.ndim == 2:
            # taken from create_extended_attention_mask_for_decoder in transformers/modeling_utils.py
            # [1...n]
            termids = torch.arange(terms_len, device=attention_mask_go_emb.device)
            # lower triangular matrix broadcasted to [batch_size, seq_len, seq_len]
            if self.config.decoder:
                causal_mask = rearrange(termids, 't -> 1 1 t') <= rearrange(termids, 't -> 1 t 1')
                causal_mask = repeat(causal_mask, '1 t1 t2 -> b t1 t2', b=batch_size)
                causal_mask = causal_mask.to(attention_mask_go_emb.dtype)
                # [batch_size, 1, seq_len, seq_len] * [batch_size, 1, 1, seq_len] -> [batch_size, 1, seq_len, seq_len] to be later broadcasted to [batch_size, num_heads, seq_len, seq_len]
                attention_mask_go_emb = rearrange(causal_mask, 'b t1 t2 -> b 1 t1 t2') * rearrange(attention_mask_go_emb, 'b t -> b 1 1 t')
            else:
                attention_mask_go_emb = repeat(attention_mask_go_emb, 'b s -> b 1 (k 1) s', k=terms_len)

            attention_mask_go_emb = attention_mask_go_emb.to(dtype=go_emb.dtype)
            attention_mask_go_emb = (1.0 - attention_mask_go_emb) * torch.finfo(go_emb.dtype).min

        # Encode sequence
        encoder_outputs = self.encoder(
            hidden_states=seq_hidden,
            attention_mask=attention_mask_seq_emb,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state


        # Decode GO terms with cross-attention to sequence
        decoder_outputs = self.decoder(
            hidden_states=go_hidden,
            attention_mask=attention_mask_go_emb,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask_seq_emb,
            return_dict=True
        )
        decoder_hidden_states = decoder_outputs.last_hidden_state

        # Generate predictions
        predictions = self.output_head(decoder_hidden_states)

        return predictions