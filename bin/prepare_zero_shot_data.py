import click
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.progress import track
from typing import Counter
import sys
sys.path.append(".")
from util import Ontology

console = Console()

# Experimental evidence codes
EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

# Terms to filter out for zero-shot evaluation (from DeepGOZero paper)
DEEPGOZERO_FILTERED_TERMS = [
    "GO:0001227",
    "GO:0001228",
    "GO:0003735",
    "GO:0004867",
    "GO:0005096",
    "GO:0000381",
    "GO:0032729",
    "GO:0032755",
    "GO:0032760",
    "GO:0046330",
    "GO:0051897",
    "GO:0120162",
    "GO:0005762",
    "GO:0022625",
    "GO:0042788",
    "GO:1904813"
]

def get_ontology_file(go_date: str, edition: str):
    """Get the specific GO ontology file path"""
    ontology_file = Path("ontologies") / f"go-{edition}-{go_date}.obo"

    if not ontology_file.exists():
        console.log(f"[red]Error: Ontology file {ontology_file} not found.[/red]")
        console.log(f"[yellow]Run: python bin/generate_go_embs.py --go-date {go_date} --edition {edition}[/yellow]")
        sys.exit(1)

    console.log(f"Using ontology file: {ontology_file}")
    return ontology_file

def propagate_annotations(annotations, go, filtered_terms):
    """
    Propagate annotations while excluding filtered terms.
    This creates the 'actual zero-shot' annotations by removing
    the specified filtered terms from the annotation hierarchy.
    """
    propagated_annotations = set()
    for annotation in annotations:
        if annotation in filtered_terms:
            continue
        propagated_annotations |= {annotation}
        propagated_annotations |= set(go.get_ancestors(annotation))
    return list(propagated_annotations)

def calculate_statistics(df):
    """Calculate and display statistics about zero-shot annotation filtering"""
    prots = []
    for tuple in df.itertuples():
        actual_set = set(tuple.actual_zero_annotations)
        prop_set = set(tuple.prop_annotations)
        if len(actual_set.symmetric_difference(prop_set)) > 0:
            prots.append((tuple.Index, actual_set - prop_set))

    if prots:
        avg_removed = sum(len(x[1]) for x in prots) / len(prots)
        console.log(f"[cyan]Zero-shot statistics:[/cyan]")
        console.log(f"  - {len(prots)} proteins have zero-shot pruned annotations")
        console.log(f"  - {avg_removed:.2f} annotations removed on average per protein")
    else:
        console.log("[yellow]No annotations were pruned[/yellow]")

def get_excluded_terms(df, go, zero_colname="zero_annotations"):
    """Get the set of terms that are excluded in zero-shot evaluation"""
    zero_cnt = Counter()
    cnt = Counter()

    for tuple in df.itertuples():
        zero_cnt.update(getattr(tuple, zero_colname))
        cnt.update(tuple.prop_annotations)

    excluded_terms = set()
    for term in go.ont:
        if term not in zero_cnt and term in cnt:
            excluded_terms.add(term)

    return excluded_terms, cnt, zero_cnt

def process_subontology(subontology: str, go: Ontology, verbose: bool = True):
    """Process train and validation data for a specific subontology"""
    data_path = Path("datasets") / "deepgozero" / subontology

    if not data_path.exists():
        console.log(f"[red]Error: {data_path} does not exist[/red]")
        return False

    console.log(f"\n[bold cyan]Processing {subontology.upper()} subontology[/bold cyan]")

    # Load data
    train_file = data_path / "train_data.pkl"
    valid_file = data_path / "valid_data.pkl"

    if not train_file.exists() or not valid_file.exists():
        console.log(f"[red]Error: Missing data files in {data_path}[/red]")
        return False

    console.log("Loading data files...")
    df_train = pd.read_pickle(train_file)
    df_valid = pd.read_pickle(valid_file)

    console.log(f"Loaded {len(df_train)} training and {len(df_valid)} validation proteins")

    # Create actual_zero_annotations column
    console.log("Creating actual_zero_annotations column...")
    df_train['actual_zero_annotations'] = df_train['exp_annotations'].apply(
        lambda x: propagate_annotations(x, go, DEEPGOZERO_FILTERED_TERMS)
    )
    df_valid['actual_zero_annotations'] = df_valid['exp_annotations'].apply(
        lambda x: propagate_annotations(x, go, DEEPGOZERO_FILTERED_TERMS)
    )

    if verbose:
        # Calculate and display statistics
        calculate_statistics(df_train)

        # Check excluded terms
        if 'zero_annotations' in df_train.columns:
            excluded_terms, _, _ = get_excluded_terms(df_train, go, "zero_annotations")
            our_excluded_terms, _, _ = get_excluded_terms(df_train, go, "actual_zero_annotations")

            console.log(f"\n[cyan]Excluded terms comparison:[/cyan]")
            console.log(f"  - Original dataset excluded: {len(excluded_terms)} terms")
            console.log(f"  - Our implementation excludes: {len(our_excluded_terms)} terms")

            discrepancy = set(our_excluded_terms).symmetric_difference(set(DEEPGOZERO_FILTERED_TERMS))
            if discrepancy:
                console.log(f"  - [yellow]Discrepancy: {discrepancy}[/yellow]")
            else:
                console.log(f"  - [green]✓ Matches expected filtered terms[/green]")

    # Save processed data
    console.log("Saving processed data...")
    df_train.to_pickle(train_file)
    df_valid.to_pickle(valid_file)

    console.log(f"[green]✓ Successfully processed {subontology.upper()}[/green]")
    return True

@click.command()
@click.option('--subontology',
              type=click.Choice(['bp', 'mf', 'cc', 'all']),
              default='all',
              help='Subontology to process (bp/mf/cc/all)')
@click.option('--verbose/--no-verbose',
              default=True,
              help='Show detailed statistics')
def main(subontology: str, verbose: bool):
    """
    Prepare zero-shot evaluation data for DeepGOZero dataset.

    This script creates the 'actual_zero_annotations' column by filtering out
    the 16 GO terms specified in the DeepGOZero paper for zero-shot evaluation.
    """
    console.log("[bold]Zero-shot Data Preparation[/bold]")

    # Get and load ontology
    ontology_file = Path("datasets") / "deepgozero" / "go.obo"
    console.log("Loading Gene Ontology...")
    go = Ontology(str(ontology_file), with_rels=True)
    console.log(f"Loaded {len(go.ont)} GO terms")

    # Process subontologies
    if subontology == 'all':
        subontologies = ['bp', 'mf', 'cc']
    else:
        subontologies = [subontology]

    success_count = 0
    for sub in subontologies:
        if process_subontology(sub, go, verbose):
            success_count += 1

    console.log(f"\n[bold green]✓ Successfully processed {success_count}/{len(subontologies)} subontologies[/bold green]")

if __name__ == "__main__":
    main()

