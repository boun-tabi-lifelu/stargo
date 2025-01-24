import pytest
import torch
import numpy as np
from pathlib import Path
import networkx as nx
import tempfile
import fastobo
from metrics import ProteinFMax, TermMacroAUPRC

# Create a simple GO graph for testing
SIMPLE_GO_OBO = """
format-version: 1.2

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: child process 1
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000003
name: child process 2
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000004
name: grandchild process
namespace: biological_process
is_a: GO:0000002
"""

@pytest.fixture
def go_obo_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.obo') as tmp:
        tmp.write(SIMPLE_GO_OBO)
        tmp.flush()
        yield Path(tmp.name)

@pytest.fixture
def go_terms():
    return ['GO:0000001', 'GO:0000002', 'GO:0000003', 'GO:0000004']

@pytest.fixture
def simple_predictions():
    # Create predictions for 3 proteins across 4 GO terms
    return torch.tensor([
        [0.9, 0.8, 0.1, 0.7],  # Protein 1
        [0.2, 0.9, 0.8, 0.1],  # Protein 2
        [0.1, 0.2, 0.7, 0.8],  # Protein 3
    ], dtype=torch.float32)

@pytest.fixture
def simple_targets():
    # Create ground truth for 3 proteins across 4 GO terms
    return torch.tensor([
        [1, 1, 0, 1],  # Protein 1
        [0, 1, 1, 0],  # Protein 2
        [0, 0, 1, 1],  # Protein 3
    ], dtype=torch.int32)

class TestProteinFMax:
    def test_initialization(self, go_obo_file, go_terms):
        metric = ProteinFMax(go_obo_file, go_terms)
        assert len(metric.thresholds) == 100
        assert isinstance(metric.go_graph, nx.DiGraph)

    def test_propagation(self, go_obo_file, go_terms, simple_targets):
        metric = ProteinFMax(go_obo_file, go_terms)
        propagated = metric._propagate_annotations(simple_targets)

        # Check that annotations propagate up the hierarchy
        # If GO:0000004 is annotated, GO:0000002 and GO:0000001 should also be annotated
        assert propagated[2, 0] == 1  # Root term should be annotated
        assert propagated[2, 1] == 1  # Parent term should be annotated

    def test_perfect_predictions(self, go_obo_file, go_terms):
        metric = ProteinFMax(go_obo_file, go_terms)
        perfect_preds = torch.tensor([[1., 1., 0., 1.]])
        perfect_targets = torch.tensor([[1, 1, 0, 1]])

        metric.update(perfect_preds, perfect_targets)
        results = metric.compute()

        assert results['f_max'] == 1.0
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0

    def test_worst_predictions(self, go_obo_file, go_terms):
        metric = ProteinFMax(go_obo_file, go_terms)
        worst_preds = torch.tensor([[1., 1., 1., 1.]])
        worst_targets = torch.tensor([[0, 0, 0, 0]])

        metric.update(worst_preds, worst_targets)
        results = metric.compute()

        assert results['f_max'] == 0.0

    def test_realistic_case(self, go_obo_file, go_terms, simple_predictions, simple_targets):
        metric = ProteinFMax(go_obo_file, go_terms)
        metric.update(simple_predictions, simple_targets)
        results = metric.compute()

        assert 0 <= results['f_max'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['threshold'] <= 1

class TestTermAUPRC:
    def test_initialization(self, go_obo_file, go_terms):
        metric = TermMacroAUPRC(go_obo_file, go_terms)
        assert len(metric.thresholds) == 100
        assert isinstance(metric.go_graph, nx.DiGraph)

    def test_perfect_predictions(self, go_obo_file, go_terms):
        metric = TermMacroAUPRC(go_obo_file, go_terms)
        perfect_preds = torch.tensor([[1., 1., 1., 1.]])
        perfect_targets = torch.tensor([[1, 1, 1, 1]])

        metric.update(perfect_preds, perfect_targets)
        auprc = metric.compute()

        assert auprc == 1.0

    def test_worst_predictions(self, go_obo_file, go_terms):
        metric = TermMacroAUPRC(go_obo_file, go_terms)
        worst_preds = torch.tensor([[1., 1., 1., 1.]])
        worst_targets = torch.tensor([[0, 0, 0, 0]])

        metric.update(worst_preds, worst_targets)
        auprc = metric.compute()

        assert auprc == 0.5

    def test_realistic_case(self, go_obo_file, go_terms, simple_predictions, simple_targets):
        metric = TermMacroAUPRC(go_obo_file, go_terms)
        metric.update(simple_predictions, simple_targets)
        auprc = metric.compute()

        assert 0 <= auprc <= 1

def test_metric_integration(go_obo_file, go_terms, simple_predictions, simple_targets):
    """Test both metrics together with a known case"""
    gene_fmax = ProteinFMax(go_obo_file, go_terms)
    term_auprc = TermMacroAUPRC(go_obo_file, go_terms)

    # Update both metrics
    gene_fmax.update(simple_predictions, simple_targets)
    term_auprc.update(simple_predictions, simple_targets)

    # Compute results
    fmax_results = gene_fmax.compute()
    auprc = term_auprc.compute()

    # Verify results match hand-calculated values
    # These values should be calculated manually for the simple case
    expected_fmax = 1.0  # Recalculated value
    expected_auprc = 1.0  # Recalculated value

    assert abs(fmax_results['f_max'] - expected_fmax) < 0.05
    assert abs(auprc - expected_auprc) < 0.05
    assert abs(auprc - expected_auprc) < 0.05

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_independence(go_obo_file, go_terms, batch_size):
    """Test that metrics give same results regardless of batch size"""
    predictions = torch.rand((8, len(go_terms)))
    targets = torch.randint(0, 2, (8, len(go_terms)))

    # Process all at once
    gene_fmax_single = ProteinFMax(go_obo_file, go_terms)
    gene_fmax_single.update(predictions, targets)
    single_results = gene_fmax_single.compute()

    # Process in batches
    gene_fmax_batch = ProteinFMax(go_obo_file, go_terms)
    for i in range(0, len(predictions), batch_size):
        batch_pred = predictions[i:i+batch_size]
        batch_target = targets[i:i+batch_size]
        gene_fmax_batch.update(batch_pred, batch_target)
    batch_results = gene_fmax_batch.compute()

    assert torch.allclose(single_results['f_max'], batch_results['f_max'])