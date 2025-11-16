from dataclasses import dataclass
from typing import Dict, Optional

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, TrainConfig
from models.bert import BERTBasedModel
from models.mlp import MLP


class TrainingModel(pl.LightningModule):
    def __init__(self, model: nn.Module, training_config: TrainConfig, priors: Optional[np.ndarray] = None):
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.lr = training_config.learning_rate

        # PU loss hyperparameters
        if priors is not None:
            # The user wants per-class priors
            prior_tensor = torch.from_numpy(priors.astype(np.float32))
        else:
            prior_tensor = torch.tensor(1e-4) # fallback to scalar

        self.register_buffer("prior", prior_tensor)
        self.register_buffer("single_prior", torch.tensor(1e-4))
        self.margin = 0.0

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        predictions = self.model(
            go_emb=inputs["go_embeddings"],
            seq_emb=inputs["embeddings"],
            attention_mask_seq_emb=inputs["attention_mask"],
            attention_mask_go_emb=None,  # All GO terms are always present
        )
        return predictions.squeeze(-1)  # Remove last dimension from sigmoid output

    def loss_fn(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.training_config.loss_fn == "bce":
            return F.binary_cross_entropy_with_logits(predictions, targets)
        elif self.training_config.loss_fn == "pu":
            return self.pu_loss(predictions, targets, self.single_prior)
        elif self.training_config.loss_fn == "pu_ranking":
            return self.pu_ranking_loss(predictions, targets, self.single_prior)
        elif self.training_config.loss_fn == "pu_ranking_priors":
            return self.pu_ranking_loss(predictions, targets, self.prior)
        else:
            raise ValueError(f"Loss function {self.training_config.loss_fn} not found")

    # [bs, num_terms]
    def pu_loss(self, predictions: torch.Tensor, targets: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        # Implement PU (Positive-Unlabeled) loss
        pos_mask = (targets == 1).float()  # 1.0 where truly positive
        unl_mask = (targets != 1).float()  # 1.0 where unlabeled

        # 1) Risk of predicting positives as negative:
        p_above = -(F.logsigmoid(predictions) * pos_mask).sum() / (
            pos_mask.sum() + 1e-10
        )

        # 2) Risk of predicting positives as positive ("below" the decision boundary):
        p_below = -(F.logsigmoid(-predictions) * pos_mask).sum() / (
            pos_mask.sum() + 1e-10
        )

        # 3) Risk of predicting unlabeled as positive:
        u_below = -(F.logsigmoid(-predictions) * unl_mask).sum() / (
            unl_mask.sum() + 1e-10
        )

        margin = self.margin  # margin for non-negative risk

        # Final PU loss: prior‐weighted positive risk plus a corrective margin term
        loss = prior * p_above + F.relu(u_below - prior * p_below + margin)
        return loss
    def pu_ranking_loss_multi(self, data, labels):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        p_above = - (F.logsigmoid(preds)*pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum(dim=0) / pos_label.sum()
        u_below = - (F.logsigmoid(preds * pos_label - preds*unl_label)).sum(dim=0) / unl_label.sum()

        loss = self.priors * p_above + th.relu(u_below - self.priors*p_below + self.margin)
        loss = loss.sum()
        return loss
    def pu_ranking_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        pos_label = (targets == 1).float()
        unl_label = (targets != 1).float()

        p_above = -(F.logsigmoid(predictions) * pos_label).sum(dim=0) / (
            pos_label.sum() + 1e-10
        )
        p_below = -(F.logsigmoid(-predictions) * pos_label).sum(dim=0) / (
            pos_label.sum() + 1e-10
        )
        u_below = (
            # ranking instead of direct penalization
            -(F.logsigmoid(predictions * pos_label - predictions * unl_label)).sum(
                dim=0
            )
            / (unl_label.sum() + 1e-10)
        )
        loss = prior * p_above + F.relu(
            u_below - prior * p_below + self.margin
        )
        return loss.sum()


    def training_step(self, batch, _batch_idx):
        inputs = {
            "embeddings": batch["embeddings"],
            "attention_mask": batch["attention_mask"],
            "go_embeddings": batch["go_embeddings"],
        }
        targets = batch["annotations"]

        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets)

        self.log("train_loss", loss, batch_size=self.training_config.batch_size)
        return loss

    def validation_step(self, batch, _batch_idx):
        inputs = {
            "embeddings": batch["embeddings"],
            "attention_mask": batch["attention_mask"],
            "go_embeddings": batch["go_embeddings"],
        }
        targets = batch["annotations"]

        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets)

        self.log("val_loss", loss, batch_size=self.training_config.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.training_config.weight_decay,
        )
        return optimizer


def get_model_cls(model_name: str):
    if model_name == "bert":
        return BERTBasedModel
    elif model_name == "mlp":
        return MLP
    else:
        raise ValueError(f"Model {model_name} not found")


def get_model(model_name: str, config: Config, priors: Optional[np.ndarray] = None):
    model_cls = get_model_cls(model_name)
    return TrainingModel(
        model=model_cls(config.model),
        training_config=config.train,
        priors=priors
    )
