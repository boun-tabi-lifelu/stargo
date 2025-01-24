import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import math
from pathlib import Path
import pickle
import torch
from typing import List, Set, Dict, Tuple
import networkx as nx
import obonet

class PUGOEvaluator:
    def __init__(self, go_obo_path: str):
        """Initialize evaluator with GO graph"""
        self.go_rels = Ontology(go_obo_path, with_rels=True)
        # Fix namespace mapping to match standard GO namespace strings
        self.namespaces = {
            'bp': 'biological_process',
            'mf': 'molecular_function',
            'cc': 'cellular_component',
            # Add full name mappings for robustness
            'biological_process': 'biological_process',
            'molecular_function': 'molecular_function',
            'cellular_component': 'cellular_component'
        }
        self.roots = {
            'bp': 'GO:0008150',
            'mf': 'GO:0003674',
            'cc': 'GO:0005575',
            'biological_process': 'GO:0008150',
            'molecular_function': 'GO:0003674',
            'cellular_component': 'GO:0005575'
        }

    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        true_annotations: List[Set[str]],
        go_terms: List[str],
        subontology: str,
    ) -> Dict:
        """Calculate all evaluation metrics for PUGO predictions

        Args:
            predictions: Model prediction tensor (n_proteins x n_terms)
            true_annotations: List of sets containing true GO annotations
            go_terms: List of GO terms corresponding to prediction columns
            subontology: Subontology to evaluate (bp/mf/cc)
        """
        # Add debug prints
        print(f"Evaluating {subontology} ontology")
        print(f"Number of input GO terms: {len(go_terms)}")
        print(f"Number of true annotations: {len(true_annotations)}")

        # Convert predictions to numpy
        pred_array = predictions.cpu().numpy()

        # Get namespace and validate
        namespace = self.namespaces.get(subontology.lower())
        if namespace is None:
            raise ValueError(f"Invalid subontology: {subontology}")
        print(f"Using namespace: {namespace}")

        # Filter terms to relevant namespace
        go_set = self.go_rels.get_namespace_terms(namespace)
        if not go_set:
            raise ValueError(f"No terms found for namespace {namespace}")
        print(f"Number of terms in namespace: {len(go_set)}")

        go_set.remove(self.roots[subontology.lower()])
        print(f"Number of terms after removing root: {len(go_set)}")

        # Calculate term information content
        self.go_rels.calculate_ic(true_annotations)

        # Debug: check term overlap
        eval_terms = set(go_terms)
        namespace_overlap = eval_terms.intersection(go_set)
        print(f"Number of evaluation terms in namespace: {len(namespace_overlap)}")

        if len(namespace_overlap) == 0:
            print("WARNING: No overlap between evaluation terms and namespace terms!")
            print("Sample eval terms:", list(eval_terms)[:5])
            print("Sample namespace terms:", list(go_set)[:5])

        # Calculate average AUC
        total_n = 0
        total_sum = 0
        for i, go_id in enumerate(go_terms):
            if go_id not in go_set:
                continue
            labels = np.array([1 if go_id in annots else 0 for annots in true_annotations])
            pos_n = np.sum(labels)
            if pos_n > 0 and pos_n < len(true_annotations):
                total_n += 1
                roc_auc = self._compute_roc(labels, pred_array[:, i])
                total_sum += roc_auc

        print(f"Number of terms used for AUC calculation: {total_n}")
        avg_auc = total_sum / total_n if total_n > 0 else 0

        # Calculate Fmax and other metrics
        fmax = 0.0
        tmax = 0.0
        smin = float('inf')
        precisions = []
        recalls = []

        for t in range(0, 101):
            threshold = t / 100.0
            pred_annotations = []

            # Get predicted annotations at this threshold
            for i in range(len(true_annotations)):
                annots = set()
                for j, score in enumerate(pred_array[i]):
                    if score >= threshold and go_terms[j] in go_set:
                        annots.add(go_terms[j])
                pred_annotations.append(annots)

            # Calculate metrics
            fscore, prec, rec, s, ru, mi, _, _, avg_ic, _ = self._evaluate_annotations(
                true_annotations, pred_annotations
            )

            precisions.append(prec)
            recalls.append(rec)

            if fscore > fmax:
                fmax = fscore
                tmax = threshold
            if s < smin:
                smin = s

        # Calculate AUPR
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_idx = np.argsort(recalls)
        recalls = recalls[sorted_idx]
        precisions = precisions[sorted_idx]
        aupr = np.trapz(precisions, recalls)

        return {
            "fmax": float(fmax),
            "threshold": float(tmax),
            "smin": float(smin),
            "aupr": float(aupr),
            "avg_auc": float(avg_auc)
        }

    def _compute_roc(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute ROC AUC score"""
        fpr, tpr, _ = roc_curve(labels, preds)
        return auc(fpr, tpr)

    def _evaluate_annotations(
        self,
        real_annots: List[Set[str]],
        pred_annots: List[Set[str]]
    ) -> Tuple:
        """Evaluate prediction accuracy metrics"""
        total = 0
        p = 0.0
        r = 0.0
        wp = 0.0
        wr = 0.0
        p_total = 0
        ru = 0.0
        mi = 0.0
        avg_ic = 0.0

        for i in range(len(real_annots)):
            if len(real_annots[i]) == 0:
                continue

            tp = real_annots[i].intersection(pred_annots[i])
            fp = pred_annots[i] - tp
            fn = real_annots[i] - tp

            tpic = sum(self.go_rels.get_norm_ic(go_id) for go_id in tp)
            fpic = sum(self.go_rels.get_norm_ic(go_id) for go_id in fp)
            fnic = sum(self.go_rels.get_norm_ic(go_id) for go_id in fn)

            avg_ic += sum(self.go_rels.get_ic(go_id) for go_id in tp)
            mi += sum(self.go_rels.get_ic(go_id) for go_id in fp)
            ru += sum(self.go_rels.get_ic(go_id) for go_id in fn)

            tpn = len(tp)
            fpn = len(fp)
            fnn = len(fn)

            total += 1
            recall = tpn / (tpn + fnn) if (tpn + fnn) > 0 else 0
            r += recall

            wrecall = tpic / (tpic + fnic) if (tpic + fnic) > 0 else 0
            wr += wrecall

            if len(pred_annots[i]) > 0:
                p_total += 1
                precision = tpn / (tpn + fpn)
                p += precision
                if tpic + fpic > 0:
                    wp += tpic / (tpic + fpic)

        # Calculate final metrics
        avg_ic = (avg_ic + mi) / total if total > 0 else 0
        ru /= total if total > 0 else 0
        mi /= total if total > 0 else 0
        r /= total if total > 0 else 0
        wr /= total if total > 0 else 0

        if p_total > 0:
            p /= p_total
            wp /= p_total

        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        wf = 2 * wp * wr / (wp + wr) if (wp + wr) > 0 else 0
        s = math.sqrt(ru * ru + mi * mi)

        return f, p, r, s, ru, mi, [], [], avg_ic, wf

class Ontology:
    """GO graph implementation"""
    def __init__(self, filename: str, with_rels: bool):
        """Initialize GO graph from OBO file

        Args:
            filename: Path to GO OBO file
            with_rels: Whether to include relationships beyond is_a
        """
        self.ont = self._load_go_graph(filename)
        self.ic = {}
        self.norm_ic = {}
        # Debug print
        print(f"Loaded GO graph with {len(self.ont.nodes)} terms")

    def _load_go_graph(self, filename: str) -> nx.DiGraph:
        """Load GO graph from OBO file"""
        graph = obonet.read_obo(filename)

        # Add edges for all parent relationships
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if 'is_a' in node_data:
                parents = node_data['is_a']
                if isinstance(parents, str):
                    parents = [parents]
                for parent in parents:
                    graph.add_edge(node, parent)

        return graph

    def get_namespace_terms(self, namespace: str) -> Set[str]:
        """Get all terms in a specific namespace"""
        terms = set()
        for node, data in self.ont.nodes(data=True):
            node_namespace = data.get('namespace', '')
            if node_namespace == namespace:
                terms.add(node)
        return terms

    def calculate_ic(self, annotations: List[Set[str]]) -> None:
        """Calculate information content for all terms"""
        # Count term frequencies
        term_counts = {}
        total_annotations = len(annotations)

        for annot_set in annotations:
            for term in annot_set:
                if term in self.ont:
                    term_counts[term] = term_counts.get(term, 0) + 1

                    # Propagate counts to ancestors
                    ancestors = nx.ancestors(self.ont, term)
                    for ancestor in ancestors:
                        term_counts[ancestor] = term_counts.get(ancestor, 0) + 1

        # Calculate IC values
        max_ic = 0
        for term, count in term_counts.items():
            if count > 0:
                self.ic[term] = -math.log(count / total_annotations)
                max_ic = max(max_ic, self.ic[term])

        # Normalize IC values
        if max_ic > 0:
            for term in self.ic:
                self.norm_ic[term] = self.ic[term] / max_ic

    def get_ic(self, go_id: str) -> float:
        """Get information content value"""
        return self.ic.get(go_id, 0.0)

    def get_norm_ic(self, go_id: str) -> float:
        """Get normalized information content value"""
        return self.norm_ic.get(go_id, 0.0)