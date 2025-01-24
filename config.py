from typing import Annotated, List, Literal, Union

import toml
from pydantic import BaseModel, Discriminator, Field, TypeAdapter, field_validator


class BaseModelConfig(BaseModel):
    name: str = Field(description="Model type ('bert' or 'mlp')")
    go_input_dim: int = Field(description="Number of dimensions of the GO term input")
    seq_input_dim: int = Field(description="Number of dimensions of the sequence input")

class BertModelConfig(BaseModelConfig):
    name: Literal["bert"]
    hidden_dim: int = Field(description="Number of hidden dimensions in the model")
    # TODO: false doesn't work right now
    decoder: bool = Field(description="Whether to use a decoder in the model or encoder-only")
    train_go_embeddings: bool = Field(description="Whether to train the GO embeddings", default=False)
    num_encoder_layers: int
    num_decoder_layers: int
    num_attention_heads: int
    hidden_act: str
    layer_norm_eps: float
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    intermediate_size: int

class MLPModelConfig(BaseModelConfig):
    name: Literal["mlp"]
    hidden_layers: List[int] = Field(description="Sizes of hidden layers")
    dropout: float = Field(description="Dropout probability")
    activation: str = Field(description="Activation function to use")
    train_go_embeddings: bool = Field(description="Whether to train the GO embeddings", default=False)

class TrainConfig(BaseModel):
    # Data paths and configuration
    data_dir: str
    go_embed_file: str = Field(
        description="Path to the files containing GO embeddings as a npy file"
    )
    protein_embed_file: str
    subontology: str
    go_release: str
    order_go_terms: bool = Field(
        default=False,
        description="Whether to order GO terms in breadth-first order"
    )

    # Compute settings
    use_tpu: bool = Field(description="Whether to use TPUs")
    prepare_data: bool = Field(
        description="Whether to prepare_data on the DataModule"
    )
    dm_num_workers: int
    bf16_precision: bool = Field(
        description="Enables/disables bfloat16 precision. Only supported on new GPUs."
    )

    # Training hyperparameters
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping: int | None = Field(
        default=10,
        description="Whether to early stop the training if the validation loss does not improve for this many epochs (default: 10)"
    )
    gradient_accumulation: int | None = Field(
        default=1,
        description="Number of batches to accumulate gradients over"
    )
    wandb_run_name: str = Field("contempro-unnamed-run", description="Name of the W&B run")

class Config(BaseModel):
    model: Union[BertModelConfig, MLPModelConfig] = Field(discriminator='name')
    train: TrainConfig


def from_toml(path: str) -> Config:
    with open(path, "r") as f:
        return Config(**toml.load(f))
