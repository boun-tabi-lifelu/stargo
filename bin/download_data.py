#!/usr/bin/env python3
"""
Download and prepare datasets for STARGO training and evaluation.

This script downloads:
1. PFresGO dataset from GitHub
   - GO ontology: 2020-06-01 (go-basic.obo)
2. DeepGOZero dataset from KAUST server
   - GO ontology: 2021-11-16 (go-basic.obo)

Note:
- Residue embeddings must be generated using bin/generate_residue_embs.py
- GO embeddings must be generated using bin/generate_go_embs.py
"""

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

# PFresGO GitHub raw file base URL
PFRESGO_BASE_URL = "https://raw.githubusercontent.com/BioColLab/PFresGO/43d7abe4752a1cf8afb22cc41b349379e6018284/Datasets"

# DeepGOZero dataset URL
DEEPGOZERO_URL = "https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/data.tar.gz"

# GO ontology configurations for each dataset
GO_CONFIGS = {
    "pfresgo": {
        "date": "2020-06-01",
        "edition": "basic",
        "url": "https://release.geneontology.org/2020-06-01/ontology/go-basic.obo"
    },
    "deepgozero": {
        "date": "2021-11-16",
        "edition": "basic",
        "url": "https://release.geneontology.org/2021-11-16/ontology/go-basic.obo"
    }
}


def download_file(url: str, output_path: Path):
    """Download a file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {output_path.name}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Downloaded {output_path.name}")


def download_pfresgo_data(data_dir: Path, force: bool = False):
    """Download PFresGO dataset files from GitHub"""
    print("\n=== Downloading PFresGO dataset ===")

    pfresgo_dir = data_dir / "pfresgo"
    pfresgo_dir.mkdir(parents=True, exist_ok=True)

    pfresgo_files = {
        "nrPDB-GO_2019.06.18_train.txt": "train.txt",
        "nrPDB-GO_2019.06.18_valid.txt": "valid.txt",
        "nrPDB-GO_2019.06.18_test.txt": "test.txt",
        "nrPDB-GO_2019.06.18_test.csv": "nrPDB-GO_2019.06.18_test.csv",
        "nrPDB-GO_2019.06.18_annot.tsv": "annot.tsv",
        "nrPDB-GO_2019.06.18_sequences.fasta": "nrPDB-GO_2019.06.18_sequences.fasta",
    }

    for source_name, target_name in pfresgo_files.items():
        output_path = pfresgo_dir / target_name

        if output_path.exists() and not force:
            print(f"Skipping {target_name} (already exists)")
            continue

        url = f"{PFRESGO_BASE_URL}/{source_name}"
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Error downloading {target_name}: {e}")
            raise


def download_deepgozero_data(data_dir: Path, force: bool = False):
    """Download and extract DeepGOZero dataset from KAUST server"""
    print("\n=== Downloading DeepGOZero dataset ===")

    deepgozero_dir = data_dir / "deepgozero"
    deepgozero_tar = deepgozero_dir / "deepgozero-data.tgz"

    if deepgozero_dir.exists() and not force:
        if (deepgozero_dir / "bp" / "train_data.pkl").exists():
            print("DeepGOZero dataset already exists, skipping")
            return

    deepgozero_dir.mkdir(parents=True, exist_ok=True)

    if not deepgozero_tar.exists() or force:
        try:
            download_file(DEEPGOZERO_URL, deepgozero_tar)
        except Exception as e:
            print(f"Error downloading DeepGOZero dataset: {e}")
            print(f"\nManual download: {DEEPGOZERO_URL}")
            print(f"Save as: {deepgozero_tar}")
            raise

    if deepgozero_tar.exists():
        print("Extracting DeepGOZero dataset...")
        with tarfile.open(deepgozero_tar, 'r:gz') as tar:
            tar.extractall(deepgozero_dir)

        # Move files from 'data' subdirectory up one level
        data_subdir = deepgozero_dir / "data"
        if data_subdir.exists():
            for item in data_subdir.iterdir():
                target = deepgozero_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            data_subdir.rmdir()

        print(f"✓ Extracted to {deepgozero_dir}")


def download_go_ontology(data_dir: Path, dataset: str, force: bool = False):
    """Download GO ontology OBO file for the specified dataset"""
    config = GO_CONFIGS[dataset]

    print(f"\n=== Downloading GO ontology for {dataset} ===")
    print(f"    Version: {config['date']} ({config['edition']})")

    dataset_dir = data_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Always save as go.obo (expected by datamodule)
    output_path = dataset_dir / "go.obo"

    if output_path.exists() and not force:
        print(f"GO ontology already exists at {output_path}")
        return

    try:
        download_file(config['url'], output_path)
    except Exception as e:
        print(f"Error downloading GO ontology: {e}")
        raise




def main():
    parser = argparse.ArgumentParser(description="Download datasets for STARGO")

    parser.add_argument("--data-dir", type=Path, default=Path("datasets"),
                        help="Root directory for datasets (default: datasets/)")
    parser.add_argument("--dataset", choices=["pfresgo", "deepgozero", "both"], default="both",
                        help="Which dataset to download (default: both)")
    parser.add_argument("--skip-ontology", action="store_true",
                        help="Skip downloading GO ontology files")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist")

    args = parser.parse_args()

    # Create directories
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Download datasets
    if args.dataset in ["pfresgo", "both"]:
        download_pfresgo_data(args.data_dir, force=args.force)
        if not args.skip_ontology:
            download_go_ontology(args.data_dir, "pfresgo", force=args.force)

    if args.dataset in ["deepgozero", "both"]:
        download_deepgozero_data(args.data_dir, force=args.force)
        if not args.skip_ontology:
            download_go_ontology(args.data_dir, "deepgozero", force=args.force)

    print("\nDataset download complete!")


if __name__ == "__main__":
    main()

