import click
from pathlib import Path
import torch
import numpy as np
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer
import fastobo
import os
import gc
import sys
sys.path.append(".")

console = Console()

def clean_memory(device="cuda"):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

def setup_directories():
    """Create necessary directories if they don't exist"""
    Path("embeddings").mkdir(exist_ok=True)
    Path("ontologies").mkdir(exist_ok=True)

def download_ontology(date: str, edition: str):
    """Download GO ontology file"""
    ontology_path = Path("ontologies") / f"go-{edition}-{date}.obo"
    if not ontology_path.exists():
        console.log(f"Downloading {edition} GO ontology for {date}")
        os.system(f"wget https://release.geneontology.org/{date}/ontology/go-{edition}.obo -O {ontology_path}")
    return ontology_path

def load_ontology(ontology_path: Path):
    """Load and process ontology file"""
    console.log("Loading ontology file...")
    annots = {}
    obo_doc = fastobo.load(str(ontology_path))

    for frame in track(obo_doc, description="Processing terms"):
        if isinstance(frame, fastobo.term.TermFrame):
            go_id = str(frame.id).replace(":", "_")
            desc = None
            name = None
            for clause in frame:
                if isinstance(clause, fastobo.term.NameClause):
                    name = str(clause)
                elif isinstance(clause, fastobo.term.DefClause):
                    desc = str(clause.definition)
                    break
            annot = desc if desc else name
            if go_id not in annots or len(annots[go_id]) < len(annot):
                annots[go_id] = annot

    return annots

def generate_sbert_embeddings(annots, device="cuda", batch_size=512):
    """Generate SBERT embeddings"""
    model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb').to(device)

    keys = list(annots.keys())
    sentences = [annots[key] for key in keys]
    embeddings = []

    for i in track(range(0, len(sentences), batch_size), description="Generating SBERT embeddings"):
        clean_memory(device)
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return {key: emb for key, emb in zip(keys, embeddings)}

def run_anc2vec(input_embeddings_path: Path, ontology_path: Path, output_path: Path):
    """Run ANC2VEC training"""
    if not Path("anc2vec").exists():
        console.log("Cloning Anc2Vec repository...")
        os.system("git clone https://github.com/mmtftr/anc2vec.git")

    console.log("Running Anc2Vec training...")
    os.system(f"""
        cd anc2vec && python -m anc2vec.train.cli \
        --obo-file {ontology_path.absolute()} \
        --embeddings-path {input_embeddings_path.absolute()} \
        --output-path {output_path.absolute()} \
        --embedding-size 200 \
        --batch-size 32 \
        --epochs 100 \
        --ance-weight 1.0 \
        --name-weight 0.5 \
        --auto-weight 0.3 \
        --initial-lr 0.001 \
        --use-lr-schedule \
        --wandb-project "anc2vec" \
        --wandb-entity "mmtf"
    """)

@click.command()
@click.option('--go-date', required=True, help='GO edition date (YYYY-MM-DD format)')
@click.option('--edition', type=click.Choice(['basic', 'full']), default='basic', help='GO edition type')
@click.option('--skip-sbert', is_flag=True, help='Skip SBERT embedding generation')
@click.option('--skip-anc2vec', is_flag=True, help='Skip ANC2VEC training')
@click.option('--device', type=click.Choice(['cuda', 'cpu', 'mps']), default='cuda', help='Compute device')
def main(go_date: str, edition: str, skip_sbert: bool, skip_anc2vec: bool, device: str):
    """Generate GO term embeddings using SBERT and ANC2VEC"""
    setup_directories()

    # Setup paths
    ontology_path = download_ontology(go_date, edition)
    sbert_output = Path("embeddings") / f"go-{edition}-{go_date}.sbert.npy"
    anc2vec_output = Path("embeddings") / f"go-{edition}-{go_date}.anc2vec.npy"

    if not skip_sbert:
        console.log("Starting SBERT embedding generation...")
        annots = load_ontology(ontology_path)
        embeddings = generate_sbert_embeddings(annots, device=device)
        np.save(sbert_output, embeddings)
        console.log(f"SBERT embeddings saved to {sbert_output}")

    if not skip_anc2vec and sbert_output.exists():
        console.log("Starting Anc2Vec training...")
        run_anc2vec(sbert_output, ontology_path, anc2vec_output)
        console.log(f"Anc2Vec embeddings saved to {anc2vec_output}")

if __name__ == "__main__":
    main()