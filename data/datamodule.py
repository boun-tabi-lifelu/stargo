from re import I
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import fastobo
import h5py
import lightning as pl
import networkx as nx
import numpy as np
import pandas as pd
from sympy import false
import torch
import wget
from goatools.obo_parser import GODag
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
from util import logger

FIXED_GO_RELEASE_OLD = "https://release.geneontology.org/2020-09-10/ontology/go-basic.obo"
GLOBAL_RESIDUE_EMB_CACHE = {}
DEFAULT_MAX_SEQ_LENGTH = 1024

@dataclass
class ProteinGoSample:
    protein_id: str
    """Data structure for a single protein-GO term sample"""

    residue_embeddings: np.ndarray
    """Residue embeddings, shape: (seq_len, seq_emb_size) dtype=float"""

    go_annotations: np.ndarray
    """GO annotations, (num_go_terms,) dtype=int"""

    go_embeddings: np.ndarray
    """GO embeddings, (num_go_terms, go_emb_size) dtype=float"""

    sequence: Optional[str] = None
    """AA sequence, optional"""

class LazyH5Dict:
    """Dictionary-like object that lazily loads embeddings from H5 file"""
    def __init__(self, h5_path: Path, is_pugo: bool = False):
        self.h5_path = h5_path
        self.is_pugo = is_pugo  # Flag to handle different protein ID formats

    def __getitem__(self, protein_id: str) -> np.ndarray:
        with h5py.File(self.h5_path, 'r') as f:
            return f[protein_id][:1024]  # Truncate to max length

    def __contains__(self, protein_id: str) -> bool:
        with h5py.File(self.h5_path, 'r') as f:
            return protein_id in f

class PFresGODataset(Dataset):
    protein_ids: List[str]
    ontologies: List[str]
    annotations: Dict[str, np.ndarray]
    go_embeddings: Dict[str, np.ndarray]
    propagate_annots: bool = False

    def __init__(
        self,
        protein_ids: List[str],
        residue_emb_file: Path,
        go_emb_file: Path | None, # None if should give token indices instead of embeddings
        ontology_file: Path, # OBO file
        annot_file: Path,
        subontology: Literal['molecular_function', 'biological_process', 'cellular_component', 'all'],
        order_go_terms: bool = False,
        propagate_annots: bool = False,
    ):
        """
        Args:
            protein_ids: List of protein IDs to include in dataset
            residue_emb_file: Path to H5 file containing residue embeddings
            go_emb_file: Path to NPY file containing GO term embeddings
            ontology_file: Path to OBO file containing ontology information
            annot_file: Path to TSV file containing GO annotations
            subontology: Subontology to include
            order_go_terms: Whether to order GO terms in breadth-first order
            propagate_annots: Whether to propagate annotations to parent terms
        """
        self.protein_ids = protein_ids

        ontology = self._load_ontology(ontology_file)

        self.subontology = subontology

        # Get GO terms list
        if order_go_terms and subontology != 'all':
            self.go_term_list = self._generate_go_terms_bfs(
                ontology_file,
                subontology
            )
        else:
            # Original implementation
            if hasattr(self, 'go_term_list'):
                logger.info(f"Using pre-existing GO term list of length {len(self.go_term_list)}")
            else:
                self.go_term_list = [
                    str(term.id) for term in ontology
                    if str(term[1].namespace) == self.subontology or self.subontology == 'all'
                ]

        self.go_lookup_table = {go_term: i for i, go_term in enumerate(self.go_term_list)}

        if go_emb_file is not None:
            # Load GO embeddings
            go_emb_dict = {k.replace('_', ':'): v for k, v in np.load(go_emb_file, allow_pickle=True).item().items()} # compatibility with GO_0000000 format

            missing_terms = list(go_term for go_term in self.go_term_list if go_term not in go_emb_dict)
            if len(missing_terms) > 0:
                logger.warning(f"{len(missing_terms)} GO term(s) {missing_terms[:3]}... not found in GO embeddings. Imputing with zero embeddings.")
                if len(missing_terms) > 0.1 * len(self.go_term_list):
                    logger.error(f"Aborting. More than 10% of all GO term(s) embeddings are missing. {len(missing_terms)}/{len(self.go_term_list)} not found in GO embeddings, please check the GO embeddings file.")
                # raise ValueError(f"Aborting. GO term(s) {missing_terms} not found in GO embeddings, please check the GO embeddings file.")

            go_emb_dim = go_emb_dict[list(go_emb_dict.keys())[0]].shape[0]

            self.go_embeddings = np.array([go_emb_dict[go_term] if go_term in go_emb_dict else np.zeros(go_emb_dim) for go_term in self.go_term_list]).astype(np.float32)
        else:
            self.go_embeddings = np.array(range(len(self.go_term_list))) # indices that will be mapped to embeddings by the model

        # Load annotations
        self.annotations = self._load_annotations(annot_file)

        # Calculate per-class priors from annotations
        self.priors = np.mean(np.stack(list(self.annotations.values())), axis=0).astype(np.float32)

        # dict of protein -> residue embeddings
        self.residues = self._load_residue_embeddings(residue_emb_file)

    def _load_residue_embeddings(self, residue_emb_file: Path) -> LazyH5Dict:
        if not residue_emb_file.exists():
            raise FileNotFoundError(f"Residue embeddings file not found: {residue_emb_file}")
        return LazyH5Dict(residue_emb_file)

    def _load_ontology(self, ontology_file: Path) -> fastobo:
        return fastobo.load(ontology_file.open('rb'))

    def _load_annotations(self, annot_file: Path) -> pd.DataFrame:
        """Load and process GO annotations from TSV file"""

        # Read TSV with pandas
        df = pd.read_csv(annot_file, sep='\t', skiprows=12).fillna('')
        df.columns = ['protein_id', 'mf', 'bp', 'cc']
        df.set_index('protein_id', inplace=True)

        # Create mapping of protein -> binary vector

        sub_short = ''.join(word[0] for word in self.subontology.split('_'))
        protein_annots_map = {}
        for protein in self.protein_ids:
            if self.subontology != 'all':
                # molecular_function -> mf, etc.

                protein_annots = df.loc[protein, sub_short].split(',')
            else:
                # Joins all annotations by comma then splits by comma
                annots = ','.join(df.loc[protein, :].values[1:])
                protein_annots = annots.split(',')

            protein_annots = [x for x in protein_annots if x != '' and x in self.go_lookup_table]
            protein_annots = np.array([self.go_lookup_table[go_term] for go_term in protein_annots], dtype=int)

            # Create binary vector
            binary_vector = np.zeros(len(self.go_term_list))
            binary_vector[protein_annots] = 1

            protein_annots_map[protein] = binary_vector

        return protein_annots_map

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> ProteinGoSample:
        protein_id = self.protein_ids[idx]

        # Get residue embeddings
        res_emb = self.residues[protein_id]

        # Get annotations for each ontology
        go_annots = self.annotations[protein_id]

        return ProteinGoSample(
            protein_id=protein_id,
            residue_embeddings=res_emb,
            go_annotations=go_annots,
            go_embeddings=self.go_embeddings
        )

    def _generate_go_terms_bfs(self, obo_file: Path, target_namespace: Literal['biological_process', 'molecular_function', 'cellular_component']) -> List[str]:
        """Generate GO terms in breadth-first order, including disconnected terms."""
        doc = fastobo.load(obo_file.open('rb'))

        # Map terms to namespaces
        term_namespaces = {}
        for frame in doc:
            if isinstance(frame, fastobo.term.TermFrame):
                term_id = str(frame.id)
                for clause in frame:
                    if isinstance(clause, fastobo.term.NamespaceClause):
                        term_namespaces[term_id] = str(clause.namespace)
                        break

        # Define roots and target namespace
        root_terms = {
            "biological_process": "GO:0008150",
            "molecular_function": "GO:0003674",
            "cellular_component": "GO:0005575"
        }
        root_term_id = root_terms[target_namespace]

        # Build graph
        graph = nx.DiGraph()
        all_terms = set()  # Keep track of all terms in the namespace

        for frame in doc:
            if isinstance(frame, fastobo.term.TermFrame):
                term_id = str(frame.id)
                if term_namespaces.get(term_id) != target_namespace:
                    continue

                all_terms.add(term_id)  # Add to all terms set
                graph.add_node(term_id)

                for clause in frame:
                    if isinstance(clause, fastobo.term.IsAClause):
                        parent_id = str(clause.term)
                        if term_namespaces.get(parent_id) == target_namespace:
                            graph.add_edge(parent_id, term_id)
                    elif isinstance(clause, fastobo.term.RelationshipClause):
                        if str(clause.typedef) == 'part_of':
                            parent_id = str(clause.term)
                            if term_namespaces.get(parent_id) == target_namespace:
                                graph.add_edge(parent_id, term_id)

        # BFS traversal starting from root
        bfs_queue = deque([root_term_id])
        visited = {root_term_id}
        bfs_order = []

        while bfs_queue:
            current_term_id = bfs_queue.popleft()
            bfs_order.append(current_term_id)

            for child_term_id in graph.successors(current_term_id):
                if child_term_id not in visited:
                    visited.add(child_term_id)
                    bfs_queue.append(child_term_id)

        # Add any remaining terms that weren't reached in BFS
        remaining_terms = all_terms - visited
        bfs_order.extend(sorted(remaining_terms))  # Add remaining terms in sorted order
        logger.info(f"Out of graph terms: {len(remaining_terms)}. These are appended to the end of the GO term list.")

        if hasattr(self, 'go_term_list'): # In the case of PUGO, we need to map the BFS order to the GO term list
            logger.info(f"Ordering pre-existing GO term list of length {len(self.go_term_list)}")
            existing_terms = set(self.go_term_list)
            if not existing_terms.issubset(set(bfs_order)):
                logger.warning(f"{len(existing_terms - set(bfs_order))} terms in the BFS order are not in the GO term list. This should not happen.")

            dropped_terms = len(set(bfs_order) - existing_terms)
            if dropped_terms > 0:
                logger.info(f"{dropped_terms} terms have been dropped according to pre-existing GO term list.")

            return [term for term in bfs_order if term in existing_terms]

        return bfs_order

class PUGODataset(PFresGODataset):
    def __init__(
        self,
        data_split_file: Path,  # Path to train/valid/test pickle file
        residue_emb_file: Path,
        go_emb_file: Path | None,
        ontology_file: Path,
        subontology: Literal['mf', 'bp', 'cc', 'all'],
        order_go_terms: bool = False,
        annotations_column: str = "prop_annotations",
    ):
        """
        Args:
            data_split_file: Path to pickle file containing split data
            residue_emb_file: Path to H5 file containing residue embeddings
            go_emb_file: Path to NPY file containing GO term embeddings
            ontology_file: Path to OBO file containing ontology information
            subontology: Subontology to include (using PU-GO's format)
            order_go_terms: Whether to order GO terms in breadth-first order
        """
        # Map PU-GO subontology format to PFresGO format
        sub_map = {
            'mf': 'molecular_function',
            'bp': 'biological_process',
            'cc': 'cellular_component',
            'all': 'all'
        }
        subontology = sub_map[subontology]

        # Load split data
        self.split_df = pd.read_pickle(data_split_file)
        protein_ids = self.split_df.proteins.values.tolist()

        # Load terms from terms.pkl in same directory as split file
        terms_df = pd.read_pickle(data_split_file.parent / "terms.pkl")
        self.go_term_list = terms_df['gos'].values.flatten()

        self.annotations_column = annotations_column

        # Initialize parent class
        super().__init__(
            protein_ids=protein_ids,
            residue_emb_file=residue_emb_file,
            go_emb_file=go_emb_file,
            ontology_file=ontology_file,
            annot_file=data_split_file,  # Not used but required by parent
            subontology=subontology,
            order_go_terms=order_go_terms
        )

    def _load_annotations(self, annot_file: Path) -> Dict[str, np.ndarray]:
        """Override parent method to load propagated annotations from PU-GO's pickle format"""
        protein_annots_map = {}

        for row in self.split_df.itertuples():
            # Create binary vector
            binary_vector = np.zeros(len(self.go_term_list))

            # Get propagated annotations if they exist
            if hasattr(row, self.annotations_column):
                annots = getattr(row, self.annotations_column)
            else:
                logger.error(f"No annotations found for protein {row.proteins}")
                annots = []

            # Set annotations in binary vector
            for go_term in annots:
                if go_term in self.go_lookup_table:
                    binary_vector[self.go_lookup_table[go_term]] = 1

            protein_annots_map[row.proteins] = binary_vector

        logger.info(f"Using the column {self.annotations_column} for annotations")
        return protein_annots_map

    def _load_residue_embeddings(self, residue_emb_file: Path) -> LazyH5Dict:
        if not residue_emb_file.exists():
            raise FileNotFoundError(f"Residue embeddings file not found: {residue_emb_file}")
        return LazyH5Dict(residue_emb_file, is_pugo=True)

class PFresGODataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        ontology: Literal['molecular_function', 'biological_process', 'cellular_component', 'all'] = 'all',
        go_release: Literal['2020', '2024'] = '2024',
        go_emb_file: str = "ontology.embeddings.npy",
        protein_emb_file: str = "per_residue_embeddings.h5",
        order_go_terms: bool = False,
        train_go_embeddings: bool = False,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ):
        """
        Args:
            data_dir: Directory to store data
            batch_size: Batch size
            num_workers: Number of workers for dataloading
            ontology: Subontology to include
            go_release: GO release to use -- only used for downloading data
            go_emb_file: GO embeddings file name in data directory
            protein_emb_file: Protein embeddings file name in data directory
            order_go_terms: Whether to order GO terms in breadth-first order
            train_go_embeddings: Whether to train the GO embeddings
            max_seq_length: Maximum sequence length for padding/truncation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ontology = ontology
        self.go_release = go_release
        self.go_emb_file = go_emb_file
        self.protein_emb_file = protein_emb_file
        self.order_go_terms = order_go_terms
        self.train_go_embeddings = train_go_embeddings
        self.max_seq_length = max_seq_length

    def prepare_data(self):
        # Use bin/download_data.py instead.
        return

    def setup(self, stage: Optional[str] = None):
        # Define file paths
        residue_emb_file = self.data_dir / "per_residue_embeddings.h5"

        ontology_file = self.data_dir / "go.obo"

        go_emb_file = self.data_dir / self.go_emb_file

        if self.train_go_embeddings:
            go_emb_file = None

        annot_file = self.data_dir / "annot.tsv"

        # Load protein IDs for each split

        # Create datasets
        if stage == "fit" or stage is None:
            train_ids = self._load_protein_ids(self.data_dir / "train.txt")
            val_ids = self._load_protein_ids(self.data_dir / "valid.txt")
            self.train_dataset = PFresGODataset(
                train_ids, residue_emb_file, go_emb_file, ontology_file,
                annot_file, self.ontology, self.order_go_terms
            )
            self.val_dataset = PFresGODataset(
                val_ids, residue_emb_file, go_emb_file, ontology_file,
                annot_file, self.ontology, self.order_go_terms
            )

        if stage == "test" or stage is None:
            test_ids = self._load_protein_ids(self.data_dir / "test.txt")
            self.test_dataset = PFresGODataset(
                test_ids, residue_emb_file, go_emb_file, ontology_file,
                annot_file, self.ontology, self.order_go_terms
            )

    @staticmethod
    def _load_protein_ids(filename: Path) -> List[str]:
        """Load protein IDs from a text file"""
        with open(filename) as f:
            return [line.strip() for line in f]

    def collate_fn(self, batch: List[ProteinGoSample]):
        if len(batch) == 0:
            logger.warning("Empty batch, skipping")
            return {}

        # Handle go_embeddings based on its shape
        go_embeddings = batch[0].go_embeddings
        if len(go_embeddings.shape) == 1:
            batch_go_embeddings = torch.from_numpy(go_embeddings).unsqueeze(0).expand(len(batch), -1)
        else:
            batch_go_embeddings = torch.from_numpy(go_embeddings).unsqueeze(0).expand(len(batch), -1, -1)
        batch_go_embeddings = batch_go_embeddings.clone()

        # Use fixed sequence length
        seq_emb_dim = batch[0].residue_embeddings.shape[1]
        batch_embeddings = torch.zeros(len(batch), self.max_seq_length, seq_emb_dim)
        batch_annotations = torch.zeros(len(batch), len(batch[0].go_annotations))
        attention_mask = torch.zeros(len(batch), self.max_seq_length)
        batch_ids = []

        for i, sample in enumerate(batch):
            # Truncate or pad to max_seq_length
            length = min(self.max_seq_length, sample.residue_embeddings.shape[0])
            batch_embeddings[i, :length] = torch.from_numpy(sample.residue_embeddings[:length])
            attention_mask[i, :length] = 1
            batch_annotations[i] = torch.from_numpy(sample.go_annotations)
            batch_ids.append(sample.protein_id)

        return {
            'embeddings': batch_embeddings,
            'annotations': batch_annotations,
            'protein_ids': batch_ids,
            'attention_mask': attention_mask,
            'go_embeddings': batch_go_embeddings
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

class PUGODataModule(PFresGODataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        ontology: Literal['molecular_function', 'biological_process', 'cellular_component'] = 'biological_process',
        go_release: Literal['2020', '2024'] = '2024',
        go_emb_file: str = "ontology.embeddings.npy",
        protein_emb_file: str = "per_residue_embeddings.h5",
        order_go_terms: bool = False,
        train_go_embeddings: bool = False,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        annotations_column: str = "prop_annotations",
    ):
        self.annotations_column = annotations_column
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            ontology=ontology,
            go_release=go_release,
            go_emb_file=go_emb_file,
            protein_emb_file=protein_emb_file,
            order_go_terms=order_go_terms,
            train_go_embeddings=train_go_embeddings,
            max_seq_length=max_seq_length,
        )

    def setup(self, stage: Optional[str] = None):
        """Override parent method to use PU-GO dataset class and file paths"""
        # Define file paths
        residue_emb_file = self.data_dir / self.protein_emb_file
        ontology_file = self.data_dir / "go.obo"

        ontology_map = {
            'molecular_function': 'mf',
            'biological_process': 'bp',
            'cellular_component': 'cc'
        }

        if self.train_go_embeddings:
            go_emb_file = None
        # we haven't performed ablation of go versions on other datasets (PUGO, NETGO)
        else:
            go_emb_file = self.data_dir / self.go_emb_file

        print(f"Using GO embeddings from: {go_emb_file}")

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = PUGODataset(
                self.data_dir / ontology_map[self.ontology] / "train_data.pkl",
                residue_emb_file, go_emb_file, ontology_file,
                ontology_map[self.ontology], self.order_go_terms, self.annotations_column
            )
            self.val_dataset = PUGODataset(
                self.data_dir / ontology_map[self.ontology] / "valid_data.pkl",
                residue_emb_file, go_emb_file, ontology_file,
                ontology_map[self.ontology], self.order_go_terms, self.annotations_column
            )

        if stage == "test" or stage is None:
            if self.data_dir.name == "pugo":
                test_filename = "time_data_esm.pkl"
            else:
                test_filename = "test_data.pkl"

            self.test_dataset = PUGODataset(
                self.data_dir / ontology_map[self.ontology] / test_filename,
                residue_emb_file, go_emb_file, ontology_file,
                ontology_map[self.ontology], self.order_go_terms, annotations_column="prop_annotations" # for test, we don't use filtered annotations
            )
