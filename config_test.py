import sys
import pytest
from pathlib import Path
import tempfile
import toml

from config import BertModelConfig, TrainConfig, from_toml, Config

# Default configurations
DEFAULT_MODEL_CONFIG = {
    "name": "bert",
    "hidden_dim": 768,
    "go_input_dim": 200,
    "seq_input_dim": 768,
    "decoder": True,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "num_attention_heads": 8,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "intermediate_size": 1024
}

DEFAULT_TRAIN_CONFIG = {
    "wandb_run_name": "test-run",
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_epochs": 100,
    "bf16_precision": False,
    "go_embed_file": "data/go_embeddings.npy",
    "protein_embed_file": "data/protein_embeddings.npy",
    "data_dir": "data",
    "subontology": "mf",
    "go_release": "2023-03-01",
    "prepare_data": False,
    "use_tpu": False,
    "dm_num_workers": 0
}

DEFAULT_CONFIG = {
    "model": DEFAULT_MODEL_CONFIG,
    "train": DEFAULT_TRAIN_CONFIG
}

def test_bert_model_config_creation():
    config = BertModelConfig(**DEFAULT_MODEL_CONFIG)
    assert config.name == "bert"
    assert config.hidden_dim == 768
    assert config.num_encoder_layers == 6
    assert config.num_attention_heads == 8
    assert config.go_input_dim == 200
    assert config.seq_input_dim == 768
    assert config.intermediate_size == 1024

def test_train_config_creation():
    config = TrainConfig(**DEFAULT_TRAIN_CONFIG)
    assert config.batch_size == 32
    assert config.learning_rate == 1e-4
    assert config.subontology == "mf"
    assert config.prepare_data is False

def test_from_toml():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as tmp:
        toml.dump(DEFAULT_CONFIG, tmp)
        tmp.flush()

        config = from_toml(tmp.name)
        assert isinstance(config, Config)
        assert isinstance(config.model, BertModelConfig)
        assert isinstance(config.train, TrainConfig)
        assert config.model.hidden_dim == 768
        assert config.train.batch_size == 32

def test_invalid_config():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BertModelConfig(hidden_dim="invalid")  # Should be int

    invalid_config = DEFAULT_CONFIG.copy()
    invalid_config["model"]["hidden_dim"] = "invalid"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as tmp:
        toml.dump(invalid_config, tmp)
        tmp.flush()

        with pytest.raises(ValidationError):
            from_toml(tmp.name)


def test_all_configs_in_configs_are_valid():
    assert Path("configs").exists()
    assert Path("configs").is_dir()
    assert len(list(Path("configs").glob("*.toml"))) > 0

    for config_file in Path("configs").glob("*.toml"):
        config = from_toml(config_file)
        assert config is not None
        assert config.model is not None
        assert config.train is not None
