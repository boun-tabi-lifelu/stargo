"""
Generate ProtT5 embeddings for protein sequences

A significant amount of the code is adapted from the PFresGO codebase (https://github.com/BioColLab/PFresGO/blob/43d7abe4752a1cf8afb22cc41b349379e6018284/fasta-embedding.py)
"""
import os
# Set environment variables before importing any deep learning libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
# Prevent TensorFlow from being loaded by transformers
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

from pathlib import Path
import pandas as pd
import argparse
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
from tqdm import tqdm
import wandb
import sys

MAX_SEQ_LENGTH = 1024 # this is the max sequence length for the downstream STARGO model, we don't need to save the embeddings after this length
# note that the ProtT5 model is given up to 4000 residues to enable the extraction of more information from the protein sequence

try:
  from IPython.core import ultratb
  sys.excepthook = ultratb.FormattedTB(mode='Plain', call_pdb=1)
except ImportError:
  print("IPython not found, using default exception hook")
  def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

  sys.excepthook = info

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.makedirs("/tmp/triton_cache", exist_ok=True)
    model = torch.compile(model)
    return model, tokenizer

def generate_and_save_embs(model, tokenizer, seqs, output_file, max_residues=4000, max_seq_len=1000, max_batch=100):
    """
    Process embeddings in batches and save to disk while generating (reduces memory usage)

    Code has been adapted from the PFresGO codebase (https://github.com/BioColLab/PFresGO/blob/43d7abe4752a1cf8afb22cc41b349379e6018284/fasta-embedding.py)
    """
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    batch = list()

    # Open H5 file in append mode
    with h5py.File(str(output_file), "a") as hf:
        with tqdm(total=len(seq_dict), desc="Generating embeddings") as pbar:
            for seq_idx, (prot_id, seq) in enumerate(seq_dict, 1):
                # Truncate sequence if longer than 4000
                if len(seq) > 4000:
                    print(f"Truncating {prot_id} from {len(seq)} to 4000 residues")
                    seq = seq[:4000]

                # Skip if protein already processed and not the last sequence in the dataset with nonempty batch
                if prot_id in hf:
                    pbar.update(1)
                    if not (seq_idx == len(seq_dict) and len(batch) > 0):
                        continue # if it's the last sequence, we want to flush the batch
                else:
                    # if the protein is not in the h5 file, we need to add it to the batch
                    seq_len = len(seq)
                    # ProtT5 requires spaces between amino acids
                    seq = ' '.join(list(seq))
                    batch.append((prot_id, seq, seq_len))

                n_res_batch = sum([s_len for _, _, s_len in batch])
                # Process batch if any of these conditions are met:
                # 1. Batch size reaches max_batch
                # 2. Total residues in batch reaches max_residues
                # 3. Current sequence is last in dataset
                # 4. Current sequence length exceeds max_seq_len
                if (len(batch) >= max_batch or
                    n_res_batch >= max_residues or
                    seq_len > max_seq_len or
                    seq_idx == len(seq_dict)):

                    # Unzip batch into separate lists
                    prot_ids, seqs, seq_lens = zip(*batch)
                    batch = list()

                    # Tokenize sequences
                    token_encoding = tokenizer.batch_encode_plus(
                        seqs,
                        add_special_tokens=True,
                        padding="longest"
                    )

                    # Convert to tensors and move to device
                    input_ids = torch.tensor(token_encoding['input_ids']).to(device)
                    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)


                    with torch.no_grad():
                        embedding_repr = model(input_ids, attention_mask=attention_mask)


                    # Process and save each embedding in batch
                    for batch_idx, identifier in enumerate(prot_ids):
                        # Get actual sequence length for this protein
                        seq_len = seq_lens[batch_idx]

                        # Truncate to min of actual length or MAX_SEQ_LENGTH
                        # Note: We take up to seq_len positions (but no more than MAX_SEQ_LENGTH)
                        actual_len = min(seq_len, MAX_SEQ_LENGTH)
                        emb = embedding_repr.last_hidden_state[batch_idx, :actual_len]

                        # Save to HDF5 file
                        hf.create_dataset(
                            identifier,
                            data=emb.detach().cpu().numpy().squeeze()
                        )
                        del emb  # Free memory

                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Update progress bar
                    pbar.update(len(prot_ids))
    if len(batch) > 0:
        print(f"Remaining {len(batch)} sequences at the end of the dataset")

def save_embeddings(emb_dict, out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)

def load_pugo_data(data_dir: Path, ontology: str) -> dict:
    """Load sequences from PU-GO dataset"""
    if ontology == "all":
        ontologies = ["bp", "mf", "cc"]
    else:
        ontologies = [ontology]

    all_seqs = {}
    for ont in ontologies:
        ont_dir = data_dir / ont
        train_df = pd.read_pickle(ont_dir / "train_data.pkl")
        valid_df = pd.read_pickle(ont_dir / "valid_data.pkl")
        test_df = pd.read_pickle(ont_dir / "test_data.pkl")

        for df in [train_df, valid_df, test_df]:
            for row in df.itertuples(index=True):
                all_seqs[row.proteins] = row.sequences

    return all_seqs

def load_pfresgo_data(data_dir: Path, ontology: str) -> dict:
    """Load sequences from PFresGO dataset

    PFresGO has a flat structure with:
    - nrPDB-GO_2019.06.18_sequences.fasta: all protein sequences
    - train.txt, valid.txt, test.txt: protein IDs for each split
    - annot.tsv: annotations
    """
    # First, load all sequences from FASTA file
    fasta_path = data_dir / "nrPDB-GO_2019.06.18_sequences.fasta"
    all_seqs = {}

    with open(fasta_path, 'r') as f:
        current_id = None
        current_seq = []

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    all_seqs[current_id] = ''.join(current_seq)

                # Extract protein ID (first field after >)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                current_seq.append(seq)

        # Save last sequence
        if current_id is not None:
            all_seqs[current_id] = ''.join(current_seq)

    # Now filter to only include sequences in train/valid/test splits
    split_ids = set()
    for split in ['train', 'valid', 'test']:
        split_file = data_dir / f"{split}.txt"
        with open(split_file, 'r') as f:
            for line in f:
                prot_id = line.strip()
                if prot_id:
                    split_ids.add(prot_id)

    # Filter sequences to only those in the splits
    filtered_seqs = {prot_id: seq for prot_id, seq in all_seqs.items() if prot_id in split_ids}

    return filtered_seqs

def main():
    parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings for protein sequences')

    # Required arguments
    parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--dataset-type', required=True, choices=['pugo', 'pfresgo'], help='Dataset type')
    parser.add_argument('--ontology', required=True, choices=['mf', 'bp', 'cc', 'all'], help='GO subontology')
    parser.add_argument('--output-file', required=True, help='Output H5 file path')

    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for embedding generation')
    parser.add_argument('--max-seq-len', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--max-residues', type=int, default=4000, help='Maximum residues per batch')

    # Wandb settings
    parser.add_argument('--wandb-project', default='protein-embeddings', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', help='Weights & Biases entity/username')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')

    args = parser.parse_args()

    # Initialize wandb if enabled
    if not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args)
            )
        except Exception as e:
            print(f"[WARNING] Failed to initialize wandb: {e}")
            print("[WARNING] Continuing without wandb logging...")

    data_dir = Path(args.data_dir)

    # Load sequences based on dataset type
    if args.dataset_type == 'pugo':
        all_seqs = load_pugo_data(data_dir, args.ontology)
    else:  # pfresgo
        all_seqs = load_pfresgo_data(data_dir, args.ontology)

    print(f"Total sequences to process: {len(all_seqs)}")

    # Generate embeddings
    model, tokenizer = get_T5_model()
    generate_and_save_embs(
        model,
        tokenizer,
        all_seqs,
        args.output_file,
        max_batch=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_residues=args.max_residues
    )

    print(f"Embeddings saved to {args.output_file}")

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()