"""
Embedding Evaluation Framework

This script evaluates and compares different GO term embeddings (Anc2Vec, OWL2Vec, BioBERT, GT2Vec (planned))
using various metrics and visualization techniques. Results are logged to Weights & Biases
for experiment tracking.
"""

from re import I, sub
import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Union, Tuple
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.rich import tqdm
import logging
import torch
import gc
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import yaml
import sys
from numba import jit
import torch
import numpy as np
import random
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(mode=True)

sys.path.append('..')

from owl2vec_star.lib.Evaluator import Evaluator
from rich.traceback import install
import warnings
from tqdm import TqdmExperimentalWarning, trange

install(show_locals=True)
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from rich.logging import RichHandler
from rich.console import Console

console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[ch]
)

logger = logging.getLogger(__name__)

# Create a sklearn-compatible wrapper for evaluation
class ModelWrapper:
    model: torch.jit.ScriptModule
    device: str

    def __init__(self, torch_model, device):
        self.model = torch.jit.script(torch_model).to("cpu")
        self.device = device
        self.model.eval()

    def predict_proba(self, X):
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            probs = self.transform_output(self.model(X_tensor)).numpy()
        return probs

    @torch.jit.script
    def transform_output(X):
        return torch.cat((X, X), dim=-1)

class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [200], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline."""
    project_name: str = "go-embedding-evaluation"
    base_ontology: Literal["go-basic", "go-full"] = "go-full"
    batch_size: int = 200
    num_runs: int = 5
    learning_rate: float = 0.001
    num_epochs: int = 100
    use_existing_model: bool = True
    test_size: float = 0.2
    random_seed: int = 42
    embedding_types: List[str] = field(default_factory=lambda: ["owl2vec"])
    models: List[Literal["mlp", "rf", "lr", "svm", "dt", "sgd"]] = field(default_factory=lambda: ["mlp"])
    enhanced: bool = False
    input_type: Literal["concatenate", "minus"] = "concatenate"
    device: str = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'EvalConfig':
        """Load configuration from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class EmbeddingDataset:
    """Handles loading and preprocessing of different embedding types."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    def load_embedding(self, file_path: Path) -> None:
        """
        Load embeddings from .npy file.

        Can load gt2vec, owl2vec, anc2vec, and biobert embeddings. Make sure embeddings are post-processed if there is a `name-post.ipynb` file.

        Expected embeddings file format (saved using np.save):
        ```py
        {
            "GO_0000001": np.ndarray([0.1, 0.2, ..., 0.3]),
            ...
        }
        ```
        """
        logger.info(f"Loading embeddings from {file_path}")
        self.embeddings[file_path.parent.name] = {k.replace(":", "_"): v for k, v in np.load(file_path, allow_pickle=True).item().items()}

        logger.info(f"Loaded {len(self.embeddings[file_path.parent.name])} {file_path.parent.name} embeddings")




class EmbeddingEvaluator(Evaluator):
    """Evaluator for embedding-based GO term classification."""
    embedding_type: str

    def __init__(self, config: EvalConfig, valid_samples: pd.DataFrame, test_samples: pd.DataFrame):
        self.config = config
        self.dataset = EmbeddingDataset(config)
        self.valid_samples = valid_samples
        self.test_samples = test_samples
        self.inferred_ancestors = {}
        self.table = wandb.Table(columns=["embedding_type", "model", "MRR", "Hits@1", "Hits@5", "Hits@10"], data=[])
        wandb.init(project=config.project_name, config=config.__dict__)

    def prepare_data(self, similarity_pairs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from similarity pairs."""
        train_x_list, train_y_list = [], []

        embedding_dim = next(iter(self.dataset.embeddings[self.embedding_type].values())).shape[0]

        for _, row in tqdm(similarity_pairs.iterrows(), desc="Preparing data", total=len(similarity_pairs)):
            sub, sup, label = row
            go_id = sub.split('/')[-1]
            sub_v = self.dataset.embeddings[self.embedding_type][go_id] if go_id in self.dataset.embeddings[self.embedding_type] else np.zeros(embedding_dim)

            go_id = sup.split('/')[-1]
            sup_v = self.dataset.embeddings[self.embedding_type][go_id] if go_id in self.dataset.embeddings[self.embedding_type] else np.zeros(embedding_dim)

            if not (np.all(sub_v == 0) or np.all(sup_v == 0)):
                if self.config.input_type == 'concatenate':
                    train_x_list.append(np.concatenate((sub_v, sup_v)))
                else:
                    train_x_list.append(sub_v - sup_v)
                train_y_list.append(int(label))

        train_X = np.array(train_x_list)
        train_y = np.array(train_y_list)
        logger.info(f'train_X: {train_X.shape}, train_y: {train_y.shape}')

        self.train_X = train_X
        self.train_y = train_y


        return -1

    # metric calculation from original code
    # used to check if results are consistent
    def other_calc(self, P, gt, classes, inferred_ancestors, sub):
        sorted_indexes = np.argsort(P, kind='stable')[::-1]
        sorted_classes = list()
        for j in sorted_indexes:
            if classes[j] not in inferred_ancestors[sub]:
                sorted_classes.append(classes[j])
        rank = sorted_classes.index(gt) + 1
        MRR_sum = 1.0 / rank
        hits1_sum = 1 if gt in sorted_classes[:1] else 0
        hits5_sum = 1 if gt in sorted_classes[:5] else 0
        hits10_sum = 1 if gt in sorted_classes[:10] else 0
        return MRR_sum, hits1_sum, hits5_sum, hits10_sum

    def evaluate(self, model, eva_samples: pd.DataFrame, should_log: bool = True) -> Tuple[float, float, float, float]:
        """Evaluate model performance using MRR and Hits@k metrics."""
        try:
            model.verbose = False
        except:
            pass

        @jit(nopython=True)
        def find_first(item, vec):
            """return the index of the first occurence of item in vec"""
            for i in range(len(vec)):
                if item == vec[i]:
                    return i
            return -1

        # Pre-compute all_classes and embeddings once
        all_classes = np.array(list(self.dataset.embeddings[self.embedding_type].keys()))
        all_class_v = np.array([self.dataset.embeddings[self.embedding_type][c] for c in all_classes])
        embedding_dim = all_class_v.shape[1]

        metrics = np.zeros(4)  # [MRR, hits@1, hits@5, hits@10]
        n_samples = len(eva_samples)

        # Extract GO IDs from URIs
        all_subs = [s.split('/')[-1] for s in eva_samples[0]]
        all_gts = [g.split('/')[-1] for g in eva_samples[1]]

        C = len(all_classes)
        emb = self.dataset.embeddings[self.embedding_type]
        if self.config.input_type == 'concatenate':
            X = np.zeros((C, embedding_dim * 2))
            X[:, embedding_dim:] = all_class_v
        else:
            X = np.zeros((C, embedding_dim))

        for idx in trange(len(eva_samples), desc="Evaluating"):
            sub_id = all_subs[idx]
            gt_id = all_gts[idx]

            if sub_id not in all_classes or gt_id not in all_classes:
                logger.warning(f"Sub or GT ID not in embedded classes: {sub_id} or {gt_id}")
                continue

            # Get embedding for subject
            sub_v = emb.get(sub_id, np.zeros(embedding_dim))

            # Prepare prediction input
            if self.config.input_type == 'concatenate':
                X[:, :embedding_dim] = sub_v[None, :]
            else:
                X = np.repeat(sub_v[None, :], C, axis=0) - all_class_v

            # Get predictions
            P = model.predict_proba(X)[:, 1]

            # Filter out inferred ancestors

            if sub_id in self.inferred_ancestors:
                mask = ~np.isin(all_classes, self.inferred_ancestors[sub_id])
                # assert gt_id not in self.inferred_ancestors[sub_id], "Superclass is an inferred ancestor"
            else:
                mask = np.ones(len(all_classes), dtype=bool)
            # assert gt_id in all_classes, "Ground truth is not in all classes"

            # Get sorted indices of valid predictions
            valid_probs = P[mask]
            valid_classes = all_classes[mask]
            sorted_indices = np.argsort(valid_probs, axis=0)[::-1]


            # Calculate metrics
            gt_idx = find_first(gt_id, valid_classes)
            sorted_idx = find_first(gt_id, valid_classes[sorted_indices])

            assert gt_idx != -1, "Ground truth is not in valid classes"
            assert sorted_idx != -1, "Ground truth is not in sorted indices"
            # breakpoint()
            new_metrics = np.zeros(4)
            new_metrics[0] += 1. / (sorted_idx + 1)
            new_metrics[1] += 1. if sorted_idx < 1 else 0
            new_metrics[2] += 1. if sorted_idx < 5 else 0
            new_metrics[3] += 1. if sorted_idx < 10 else 0

            metrics += new_metrics

            # Log progress periodically
            if (idx + 1) % 1000 == 0:
                logger.info(
                    f'Evaluated {idx + 1} samples - '
                    f'MRR: {metrics[0]/(idx+1)}, '
                    f'Hits@1: {metrics[1]/(idx+1)}, '
                    f'Hits@5: {metrics[2]/(idx+1)}, '
                    f'Hits@10: {metrics[3]/(idx+1)}'
                )

        # Compute final metrics
        metrics /= n_samples

        model_name = model.__class__.__name__
        prefix = f"{self.config.base_ontology}/{self.embedding_type}/{model_name}"

        # Log to wandb
        if should_log:
            self.table.add_data(self.embedding_type, model_name, metrics[0], metrics[1], metrics[2], metrics[3])
            wandb.log({
                f"{prefix}/MRR": metrics[0],
                f"{prefix}/Hits@1": metrics[1],
                f"{prefix}/Hits@5": metrics[2],
                f"{prefix}/Hits@10": metrics[3]
            })

        return tuple(metrics)

    def run_torch_mlp(self):
        """Run MLP training using PyTorch"""
        # Convert data to torch tensors
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_X, self.train_y,
            test_size=0.1,
            random_state=42
        )

        X_train = torch.FloatTensor(X_train).to(self.config.device)
        y_train = torch.FloatTensor(y_train).to(self.config.device)
        X_val = torch.FloatTensor(X_val).to(self.config.device)
        y_val = torch.FloatTensor(y_val).to(self.config.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )

        # Initialize model
        input_dim = X_train.shape[1]
        model = TorchMLP(input_dim).to(self.config.device)

        # Set up model save path
        model_dir = Path(f"{self.config.base_ontology}/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"torch_mlp_{self.embedding_type}.pt"

        # Load saved model if it exists
        if model_path.exists():
            logger.info(f"Loading saved model from {model_path}")
            model.load_state_dict(torch.load(model_path))

        if model_path.exists() and self.config.use_existing_model:
            wrapper = ModelWrapper(model, self.config.device)
            MRR, hits1, hits5, hits10 = self.evaluate(model=wrapper, eva_samples=self.test_samples)
            print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' %
                (MRR, hits1, hits5, hits10))
            return

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    val_loss += criterion(outputs, batch_y).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
                # Save best model
                torch.save(best_state, model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            # Log metrics
            prefix = f"{self.config.base_ontology}/{self.embedding_type}/torch_mlp"
            wandb.log({
                f"{prefix}/train_loss": train_loss,
                f"{prefix}/val_loss": val_loss,
                f"{prefix}/learning_rate": optimizer.param_groups[0]['lr'],
                f"{prefix}/epoch": epoch
            })

            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Load best model
        model.load_state_dict(best_state)



        wrapper = ModelWrapper(model, self.config.device)
        MRR, hits1, hits5, hits10 = self.evaluate(model=wrapper, eva_samples=self.test_samples)
        print('Testing, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n\n' %
            (MRR, hits1, hits5, hits10))


def main():
    """Main execution function."""
    # Load configuration
    config = EvalConfig.from_yaml("config.yaml")

    base_path = Path(f"{config.base_ontology}")


    # Load data
    train_data = pd.read_csv(base_path / "split" / "train.csv", header=None)  # Add header=None
    valid_data = pd.read_csv(base_path / "split" / "valid.csv", header=None)
    test_data = pd.read_csv(base_path / "split" / "test.csv", header=None)

    # Initialize evaluator
    evaluator = EmbeddingEvaluator(config, valid_data, test_data)

    # Load embeddings
    # evaluates whichever ones are loaded
    for embedding_type in config.embedding_types:
        evaluator.dataset.load_embedding(base_path / embedding_type / "ontology.embeddings.npy")

    # Validate data structure
    logger.info(f"Train data shape: {train_data.shape}")

    # Load inferred ancestors
    with open(base_path / "split" / "inferred_ancestors.txt") as f:
        evaluator.inferred_ancestors = {
            line.strip().split(',')[0].split('/')[-1]: [s.split('/')[-1] for s in line.strip().split(',')]
            for line in f.readlines()
        }

    for run in range(config.num_runs):
        # Run evaluation for each embedding type
        for embedding_type in evaluator.dataset.embeddings.keys():
            evaluator.embedding_type = embedding_type
            logger.info(f"\nEvaluating {embedding_type} embeddings")

            # Prepare data
            evaluator.prepare_data(train_data)

            # Run different models
            if "rf" in config.models:
                logger.info("\nRandom Forest:")
                evaluator.run_random_forest()

            if "mlp" in config.models:
                logger.info("\nMLP:")
                evaluator.run_mlp()

            if "torch-mlp" in config.models:
                logger.info("\nTorch MLP:")
                evaluator.run_torch_mlp()

            if "lr" in config.models:
                logger.info("\nLogistic Regression:")
                evaluator.run_logistic_regression()

            if "svm" in config.models:
                logger.info("\nSVM:")
                evaluator.run_svm()

            if "dt" in config.models:
                logger.info("\nDecision Tree:")
                evaluator.run_decision_tree()

            if "sgd" in config.models:
                logger.info("\nSGD Logistic:")
                evaluator.run_sgd_log()

            gc.collect()

    wandb.log({"summary": evaluator.table})
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
