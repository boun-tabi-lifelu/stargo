from typing import List, Optional, Dict
from pathlib import Path
import torch
from torchmetrics import Metric
import numpy as np
from tqdm import tqdm
import fastobo
import networkx as nx
from sklearn.metrics import auc, precision_recall_curve

class ProteinFunctionMetric(Metric):
    """Base class for protein function prediction metrics"""
    def __init__(self,
                 ontology_file: Path,
                 go_terms: List[str],
                 num_thresholds: int = 100):
        super().__init__()

        self.go_terms = go_terms
        self.thresholds = torch.linspace(0, 1, num_thresholds)
        self.go_graph = self._build_go_graph(ontology_file)

        # Add states for accumulation
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def _build_go_graph(self, obo_file: Path) -> nx.DiGraph:
        """Build GO graph from OBO file"""
        doc = fastobo.load(obo_file.open('rb'))
        graph = nx.DiGraph()

        # Map terms to namespaces
        term_namespaces = {}
        for frame in doc:
            if isinstance(frame, fastobo.term.TermFrame):
                term_id = str(frame.id)
                for clause in frame:
                    if isinstance(clause, fastobo.term.NamespaceClause):
                        term_namespaces[term_id] = str(clause.namespace)
                        break

        # Build graph with is_a and part_of relationships
        for frame in doc:
            if isinstance(frame, fastobo.term.TermFrame):
                term_id = str(frame.id)
                if term_id not in self.go_terms:
                    continue

                graph.add_node(term_id, aspect=term_namespaces[term_id])

                for clause in frame:
                    if isinstance(clause, fastobo.term.IsAClause):
                        parent_id = str(clause.term)
                        if parent_id in self.go_terms:
                            graph.add_edge(parent_id, term_id)
                    elif isinstance(clause, fastobo.term.RelationshipClause):
                        if str(clause.typedef) == 'part_of':
                            parent_id = str(clause.term)
                            if parent_id in self.go_terms:
                                graph.add_edge(parent_id, term_id)

        return graph


    def _propagate_annotations(self, annotations: torch.Tensor) -> torch.Tensor:
        """Propagate annotations up the GO graph"""
        # Convert annotations to boolean tensor
        propagated = annotations.clone()

        # Convert GO term indices to actual terms
        term_to_idx = {term: idx for idx, term in enumerate(self.go_terms)}

        # Vectorized propagation
        for term_idx, term in enumerate(self.go_terms):
            if term in self.go_graph:
                ancestors = nx.ancestors(self.go_graph, term)
                ancestor_indices = [term_to_idx[anc] for anc in ancestors if anc in term_to_idx]
                if ancestor_indices:
                    propagated[:, ancestor_indices] = torch.max(
                        propagated[:, ancestor_indices],
                        annotations[:, term_idx].unsqueeze(1)
                    )

        return propagated

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric states with new predictions and targets"""
        self.predictions.append(preds)
        self.targets.append(target)

    def compute(self) -> torch.Tensor:
        """Compute final metric value"""
        raise NotImplementedError


class ProteinFMax(ProteinFunctionMetric):
    """Protein-centric F-max metric"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional states for F-max computation
        self.add_state("best_f", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("best_threshold", default=torch.tensor(0.0))
        self.add_state("best_precision", default=torch.tensor(0.0))
        self.add_state("best_recall", default=torch.tensor(0.0))

    def compute(self) -> Dict[str, torch.Tensor]:
        # Concatenate accumulated predictions and targets
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)

        # Propagate annotations for both predictions and targets
        predictions = self._propagate_annotations(predictions)
        targets = self._propagate_annotations(targets)

        f_scores = []
        precisions = []
        recalls = []

        for threshold in self.thresholds:
            pred_binary = (predictions >= threshold).float()

            tp = (pred_binary * targets).sum(dim=1)
            fp = (pred_binary * (1 - targets)).sum(dim=1)
            fn = ((1 - pred_binary) * targets).sum(dim=1)

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)

            mask = pred_binary.sum(dim=1) > 0
            if mask.sum() > 0:
                avg_precision = precision[mask].mean()
                avg_recall = recall.mean()
                f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-10)
            else:
                f1 = torch.tensor(0.0)
                avg_precision = torch.tensor(0.0)
                avg_recall = torch.tensor(0.0)

            f_scores.append(f1)
            precisions.append(avg_precision)
            recalls.append(avg_recall)

        f_scores = torch.stack(f_scores)
        best_idx = torch.argmax(f_scores)

        return {
            'f_max': f_scores[best_idx],
            'threshold': self.thresholds[best_idx],
            'precision': precisions[best_idx],
            'recall': recalls[best_idx]
        }


class TermMacroAUPRC(ProteinFunctionMetric):
    """Term-centric AUPRC metric"""
    def compute(self) -> torch.Tensor:
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)

        # Propagate annotations
        predictions = self._propagate_annotations(predictions)
        targets = self._propagate_annotations(targets)

        term_auprc = []

        for i in range(predictions.shape[1]):
            term_preds = predictions[:, i]
            term_truth = targets[:, i]

            precisions, recalls, _ = precision_recall_curve(term_truth.numpy(), term_preds.numpy())

            term_auprc.append(auc(recalls, precisions))

        return torch.tensor(np.mean(term_auprc))

class TermMicroAUPRC(ProteinFunctionMetric):
    """Flattened Micro AUPRC metric"""
    def compute(self) -> torch.Tensor:
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)

        # Propagate annotations
        predictions = self._propagate_annotations(predictions)
        targets = self._propagate_annotations(targets)

        # Compute micro AUPRC
        precision, recall, _ = precision_recall_curve(targets.flatten().numpy(), predictions.flatten().numpy())

        return auc(recall, precision)

