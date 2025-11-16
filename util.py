import os
import sys
import logging
from loguru import logger
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm
import time
import random
from collections import deque, Counter
import math


# --- Configuration ---
LOG_LEVEL = "INFO"  # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# --- Jupyter/Colab Check ---
def is_in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # Check if not in Jupyter
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def get_handler():
    IN_NOTEBOOK = is_in_notebook()

    # --- Loguru + Rich Handler Setup ---
    if IN_NOTEBOOK:
        # For Jupyter/Colab, use the default RichHandler
        handler = RichHandler(
            rich_tracebacks=True,  # Show rich tracebacks for exceptions
            markup=True  # Enable rich markup
        )
    else:
        # For terminal, customize RichHandler for better formatting
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=False,  # We have time in the log format
            show_level=False,  # We have level in the format
            show_path=False,  # We have path in the format
        )
    return handler

def setup_distributed_logger(rank=None):
    """
    Configure the logger for distributed training environments.

    In distributed training with FSDP (Fully Sharded Data Parallel),
    we typically want to only log from rank 0 to avoid duplicate logs.

    Args:
        rank (int, optional): The process rank in distributed training.
            If None, will attempt to get rank from environment variables.
            If rank is not 0, logging will be disabled.
    """
    # Try to get rank from environment if not provided
    if rank is None:
        # Check common environment variables used in distributed training
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

    # Remove all existing handlers
    logger.remove()

    # Only add handlers for rank 0
    if rank == 0:
        # Add console handler with the configured format
        handler = get_handler()
        logger.add(handler, format=LOG_FORMAT, level=LOG_LEVEL, backtrace=True, diagnose=True, catch=True)
        logger.info(f"Logger configured for rank {rank} (main process)")
    else:
        # For non-zero ranks, add a minimal handler with ERROR level only
        # This ensures critical errors are still captured but normal logs are suppressed
        logger.add(
            lambda msg: None,  # No-op handler
            level="ERROR"
        )

setup_distributed_logger()

# --- Ontology Parser from DeepGOZero ---
class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)

        return ont

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_ancestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set
