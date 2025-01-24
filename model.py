from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import lightning as pl
from config import Config, TrainConfig
from models.bert import BERTBasedModel
import torch.nn.functional as F

from models.mlp import MLP


class TrainingModel(pl.LightningModule):
    def __init__(self, model: nn.Module, training_config: TrainConfig):
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.lr = training_config.learning_rate

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        predictions = self.model(
            go_emb=inputs['go_embeddings'],
            seq_emb=inputs['embeddings'],
            attention_mask_seq_emb=inputs['attention_mask'],
            attention_mask_go_emb=None  # All GO terms are always present
        )
        return predictions.squeeze(-1)  # Remove last dimension from sigmoid output

    def training_step(self, batch, _batch_idx):
        inputs = {
            'embeddings': batch['embeddings'],
            'attention_mask': batch['attention_mask'],
            'go_embeddings': batch['go_embeddings']
        }
        targets = batch['annotations']

        predictions = self(inputs)
        loss = F.binary_cross_entropy_with_logits(predictions, targets)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs = {
            'embeddings': batch['embeddings'],
            'attention_mask': batch['attention_mask'],
            'go_embeddings': batch['go_embeddings']
        }
        targets = batch['annotations']

        predictions = self(inputs)
        # TODO: add loss to training config
        loss = F.binary_cross_entropy_with_logits(predictions, targets)

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.training_config.weight_decay)
        return optimizer

def get_model_cls(model_name: str):
    if model_name == "bert":
        return BERTBasedModel
    elif model_name == "mlp":
        return MLP
    else:
        raise ValueError(f"Model {model_name} not found")

def get_model(model_name: str, config: Config):
    model_cls = get_model_cls(model_name)
    return TrainingModel(
        model=model_cls(config.model),
        training_config=config.train
    )

