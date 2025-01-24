"""
Generate ProtT5 embeddings for protein sequences

A significant amount of the code is adapted from the PFresGO codebase (https://github.com/BioColLab/PFresGO/blob/43d7abe4752a1cf8afb22cc41b349379e6018284/fasta-embedding.py)
"""
import os
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

MAX_SEQ_LENGTH = 1024 # this is the max sequence length for the downstream CONTEMPRO model, we don't need to save the embeddings after this length
# note that the ProtT5 model is given up to 4000 residues to enable the extraction of more information from the protein sequence

try:
  from IPython.core import ultratb
  sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=1)
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
                # Skip if protein already processed and not the last sequence in the dataset with nonempty batch
                if prot_id in hf:
                    pbar.update(1)
                    if not (seq_idx == len(seq_dict) and len(batch) > 0):
                        continue # if it's the last sequence, we want to flush the batch
                else:
                    # if the protein is not in the h5 file, we need to add it to the batch
                    seq_len = len(seq)
                    seq = ' '.join(list(seq))
                    batch.append((prot_id, seq, seq_len))


                # Truncate sequence if longer than 4000
                if len(seq) > 4000:
                    print(f"Truncating {prot_id} from {len(seq)} to 4000 residues")
                    seq = seq[:4000]


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

                    try:
                        # Generate embeddings without gradients
                        with torch.no_grad():
                            embedding_repr = model(input_ids, attention_mask=attention_mask)
                    except RuntimeError:
                        # Debug on error
                        import ipdb
                        ipdb.post_mortem()
                        print(f"RuntimeError during embedding for {prot_id} (L={seq_len})")
                        continue

                    # Process and save each embedding in batch
                    for batch_idx, identifier in enumerate(prot_ids):
                        # Get embedding and truncate to max sequence length
                        emb = embedding_repr.last_hidden_state[batch_idx, :MAX_SEQ_LENGTH]

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
    ont_dir = data_dir / ontology
    train_df = pd.read_pickle(ont_dir / "train_data.pkl")
    valid_df = pd.read_pickle(ont_dir / "valid_data.pkl")
    test_df = pd.read_pickle(ont_dir / "test_data.pkl")

    all_seqs = {}
    for df in [train_df, valid_df, test_df]:
        for row in df.itertuples(index=True):
            all_seqs[row.proteins] = row.sequences

    return all_seqs

def load_pfresgo_data(data_dir: Path, ontology: str) -> dict:
    """Load sequences from PFresGO dataset"""
    # Map short ontology names to PFresGO format
    ont_map = {
        'mf': 'molecular_function',
        'bp': 'biological_process',
        'cc': 'cellular_component'
    }
    ont_name = ont_map.get(ontology, ontology)

    splits = ['train', 'valid', 'test']
    all_seqs = {}

    for split in splits:
        tsv_path = data_dir / ont_name / f"{split}.tsv"
        df = pd.read_csv(tsv_path, sep='\t')
        for _, row in df.iterrows():
            all_seqs[row['protein_id']] = row['sequence']

    return all_seqs

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
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )

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