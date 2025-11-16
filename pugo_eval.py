import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import math
from pathlib import Path
import pickle
import torch
from typing import List, Set, Dict, Tuple, Optional
import networkx as nx
import obonet
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from networkx import descendants

from util import logger

class PUGOEvaluator:
    def __init__(self, go_obo_path: str):
        """Initialize evaluator with GO graph"""
        logger.info(f"Initializing PUGOEvaluator with OBO file: {go_obo_path}")
        start_time = time.time()
        self.go_rels = Ontology(go_obo_path, with_rels=True)
        logger.info(f"Loaded GO ontology in {time.time() - start_time:.2f} seconds")

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

        self.root_terms = {
            'GO:0008150',
            'GO:0003674',
            'GO:0005575'
        }

    # Propagates annotations
    # predictions [n_proteins, n_terms]
    # go_term_list [n_terms]
    # uses go_rels to propagate annotations
    def propagate_annotations(
        self,
        predictions: torch.Tensor,
        go_term_list: List[str]
    ) -> torch.Tensor:
        d = predictions.device
        # Get namespace and validate
        idx_map = {term: i for i, term in enumerate(go_term_list)}

        assert predictions.shape[-1] == len(go_term_list)

        predictions = predictions.clone()

        terms = set(go_term_list)
        for i, term in enumerate(go_term_list):
            if term not in self.go_rels.ont:
                raise ValueError(f"Term {term} not found in GO graph")
            parents = terms.intersection(descendants(self.go_rels.ont, term))

            for parent in parents:
                if parent in self.root_terms:
                    continue
                predictions[:, idx_map[parent]] = torch.max(predictions[:, idx_map[parent]], predictions[:, i])

        return predictions


    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        true_annotations: List[Set[str]],
        go_terms: List[str],
        subontology: str,
    ) -> Dict:
        """Calculate all evaluation metrics for PUGO predictions

        Args:
            predictions: Model prediction tensor (propagated) (n_proteins x n_terms)
            true_annotations: List of sets containing true and propagated GO annotations
            go_terms: List of GO terms corresponding to prediction columns
            subontology: Subontology to evaluate (bp/mf/cc)
        """
        # Add debug prints
        logger.info(f"Evaluating {subontology} ontology")
        logger.info(f"Number of input GO terms: {len(go_terms)}")

        # for debugging, test how many of the terms have any true annots
        start_time = time.time()
        terms_with_annots = set()
        for prot_annots in true_annotations:
            for term in prot_annots:
                if term in go_terms:
                    terms_with_annots.add(term)
        logger.info(f"Number of terms with any true annots: {len(terms_with_annots)} (took {time.time() - start_time:.2f}s)")

        logger.info(f"Number of proteins in true annotations: {len(true_annotations)}")

        pred_array = predictions.cpu().numpy()

        # Get namespace and validate
        namespace = self.namespaces.get(subontology.lower())
        if namespace is None:
            raise ValueError(f"Invalid subontology: {subontology}")
        logger.info(f"Using namespace: {namespace}")

        go_set = self.go_rels.get_namespace_terms(namespace)

        if not go_set:
            raise ValueError(f"No terms found for namespace {namespace}")
        logger.info(f"Number of terms in namespace: {len(go_set)}")

        go_set.remove(self.roots[subontology.lower()])

        # Calculate term information content
        self.go_rels.calculate_ic(true_annotations)

        # Debug: check term overlap
        eval_terms = set(go_terms)
        namespace_overlap = eval_terms.intersection(go_set)
        logger.info(f"Number of evaluation terms in namespace: {len(namespace_overlap)}")

        if len(namespace_overlap) == 0:
            logger.warning("WARNING: No overlap between evaluation terms and namespace terms!")
            logger.warning(f"Sample eval terms: {list(eval_terms)[:5]}")
            logger.warning(f"Sample namespace terms: {list(go_set)[:5]}")

        # Calculate average AUC
        start_time = time.time()
        total_n = 0
        total_sum = 0
        logger.info("Calculating AUC for each term...")
        for i, go_id in tqdm(enumerate(go_terms), total=len(go_terms), desc="Calculating AUC"):
            if go_id not in go_set:
                continue
            labels = np.array([1 if go_id in annots else 0 for annots in true_annotations])
            pos_n = np.sum(labels)
            if pos_n > 0 and pos_n < len(true_annotations):
                total_n += 1
                roc_auc, _, _ = self._compute_roc(labels, pred_array[:, i])
                total_sum += roc_auc

        logger.info(f"AUC calculation took {time.time() - start_time:.2f}s")
        avg_auc = total_sum / total_n if total_n > 0 else 0

        # Calculate Fmax and other metrics
        start_time = time.time()
        fmax = 0.0
        tmax = 0.0
        smin = float('inf')
        precisions = []
        recalls = []

        logger.info("Calculating Fmax and other metrics at different thresholds...")
        for t in tqdm(range(0, 101), desc="Threshold evaluation"):
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

        logger.info(f"Fmax calculation took {time.time() - start_time:.2f}s")

        # Calculate AUPR
        start_time = time.time()
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_idx = np.argsort(recalls)
        recalls = recalls[sorted_idx]
        precisions = precisions[sorted_idx]
        aupr = np.trapz(precisions, recalls)
        logger.info(f"AUPR calculation took {time.time() - start_time:.2f}s")

        return {
            "fmax": float(fmax),
            "threshold": float(tmax),
            "smin": float(smin),
            "aupr": float(aupr),
            "avg_auc": float(avg_auc)
        }

    def _compute_roc(self, labels: np.ndarray, preds: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute ROC AUC score and return FPR/TPR"""
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        return roc_auc, fpr, tpr

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

    def calculate_zeroshot_metrics(
        self,
        predictions: torch.Tensor,
        true_annotations: List[Set[str]],
        go_terms: List[str],
        subontology: str,
        filtered_terms: List[str],
        model_name: str = "model",
        save_plots_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate ROC AUC for a specific list of filtered GO terms and optionally save plots.

        Args:
            predictions: Model prediction tensor (n_proteins x n_terms)
            true_annotations: List of sets containing true GO annotations (not propagated)
            go_terms: List of GO terms corresponding to prediction columns
            subontology: Subontology being evaluated (bp/mf/cc)
            filtered_terms: List of specific GO terms to calculate AUC for.
            model_name: Name of the model, used for plot filenames.
            save_plots_dir: Directory to save ROC curve plots. If None, plots are not saved.

        Returns:
            A dictionary containing term-specific AUCs ('term_auc') and the mean AUC ('mean_auc').
        """
        logger.info(f"Evaluating zero-shot metrics for {subontology} ontology")
        logger.info(f"Number of filtered terms provided: {len(filtered_terms)}")

        pred_array = predictions.cpu().numpy()
        namespace = self.namespaces.get(subontology.lower())
        if namespace is None:
            raise ValueError(f"Invalid subontology: {subontology}")

        go_set = self.go_rels.get_namespace_terms(namespace)
        if not go_set:
            raise ValueError(f"No terms found for namespace {namespace}")

        # Create a lookup for go_terms index
        go_term_to_index = {term: i for i, term in enumerate(go_terms)}

        term_aucs = {}
        evaluated_term_count = 0

        logger.info("Calculating AUC for filtered terms...")
        for term_id in tqdm(filtered_terms, desc="Calculating Zero-Shot AUC"):
            if term_id not in go_set:
                # logger.debug(f"Skipping term {term_id}: Not in the {namespace} namespace.")
                continue
            if term_id not in go_term_to_index:
                # logger.debug(f"Skipping term {term_id}: Not found in the model's output terms.")
                continue

            term_index = go_term_to_index[term_id]
            labels = np.array([1 if term_id in annots else 0 for annots in true_annotations])
            pos_n = np.sum(labels)

            # Check if term has both positive and negative examples
            if pos_n > 0 and pos_n < len(true_annotations):
                roc_auc, fpr, tpr = self._compute_roc(labels, pred_array[:, term_index])
                term_aucs[term_id] = float(roc_auc)
                evaluated_term_count += 1

                # Save ROC plot if requested
                if save_plots_dir:
                    os.makedirs(save_plots_dir, exist_ok=True)
                    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
                    display.plot()
                    plot_filename = os.path.join(save_plots_dir, f"{model_name}_{subontology}_zero_auc_{term_id.replace(':', '-')}.png") # Replace : for filename compatibility
                    plt.title(f'ROC Curve for {term_id} ({subontology})')
                    plt.savefig(plot_filename)
                    plt.close() # Close the plot to free memory
            else:
                # logger.debug(f"Skipping term {term_id}: Not enough positive/negative examples (Pos: {pos_n}, Total: {len(true_annotations)}).")
                term_aucs[term_id] = None # Indicate that AUC could not be calculated

        logger.info(f"Calculated AUC for {evaluated_term_count} out of {len(filtered_terms)} provided terms.")
        # Filter out terms where AUC couldn't be calculated
        final_aucs = {k: v for k, v in term_aucs.items() if v is not None}

        # Calculate mean AUC
        mean_auc = np.mean(list(final_aucs.values())) if final_aucs else 0.0
        logger.info(f"Mean AUC for filtered terms: {mean_auc:.4f}")

        # results = {"term_auc": final_aucs, "mean_auc": float(mean_auc)}

        return final_aucs

class Ontology:
    """GO graph implementation"""
    def __init__(self, filename: str, with_rels: bool):
        """Initialize GO graph from OBO file

        Args:
            filename: Path to GO OBO file
            with_rels: Whether to include relationships beyond is_a
        """
        start_time = time.time()
        self.ont = self._load_go_graph(filename)
        logger.info(f"Loaded GO graph in {time.time() - start_time:.2f}s")
        self.ic = {}
        self.norm_ic = {}
        # Debug print
        logger.info(f"Loaded GO graph with {len(self.ont.nodes)} terms")

    def _load_go_graph(self, filename: str) -> nx.DiGraph:
        """Load GO graph from OBO file"""
        return obonet.read_obo(filename)

    def get_namespace_terms(self, namespace: str) -> Set[str]:
        """Get all terms in a specific namespace"""
        start_time = time.time()
        terms = set()
        for node, data in tqdm(self.ont.nodes(data=True), total=len(self.ont.nodes), desc="Filtering namespace terms"):
            node_namespace = data.get('namespace', '')
            if node_namespace == namespace:
                terms.add(node)
        logger.info(f"Filtered namespace terms in {time.time() - start_time:.2f}s")
        return terms

    def calculate_ic(self, annotations: List[Set[str]]) -> None:
        """Calculate information content for all terms (provided propagated annotations)"""
        start_time = time.time()
        # Count term frequencies
        from collections import Counter
        term_counts = Counter()

        total_annotations = len(annotations)
        logger.info("Counting term frequencies...")
        for annot_set in tqdm(annotations, desc="Processing annotations"):
            for term in annot_set:
                if term in self.ont:
                    term_counts[term] += 1

        # Calculate IC values
        logger.info("Calculating IC values...")
        max_ic = 0
        for term, count in tqdm(term_counts.items(), desc="Calculating IC"):
            # find min parent annot_count
            min_n = total_annotations

            for parent in self.ont.neighbors(term):
                if parent in term_counts:
                    min_n = min(min_n, term_counts[parent])
            if min_n > 0:
                self.ic[term] = math.log(min_n / count, 2)
                max_ic = max(max_ic, self.ic[term])

        # Normalize IC values
        if max_ic > 0:
            for term in self.ic:
                self.norm_ic[term] = self.ic[term] / max_ic

        logger.info(f"Calculated IC values in {time.time() - start_time:.2f}s")

    def get_ic(self, go_id: str) -> float:
        """Get information content value"""
        return self.ic.get(go_id, 0.0)

    def get_norm_ic(self, go_id: str) -> float:
        """Get normalized information content value"""
        return self.norm_ic.get(go_id, 0.0)