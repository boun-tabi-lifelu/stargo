# STAR-GO: Improving Protein Function Prediction by Learning to Hierarchically Integrate Ontology-Informed Semantic Embeddings

## Prerequisites

- **Environment**: Reproducible environment provided via Docker (see `Dockerfile`) or Conda (see `environment.yml`)
- **Experiment Tracking**: Weights & Biases (wandb) account for experiment tracking and artifact management (optional but recommended)

## Data Preparation

### Automated Setup (Recommended)

The easiest way to prepare all required datasets is to use the automated setup script:

```bash
./scripts/prepare_data.sh --dataset both --device cuda
```

This script will automatically:
1. Download PFresGO and DeepGOZero datasets
2. Download GO ontology files (version 2020-06-01 for PFresGO, 2021-11-16 for DeepGOZero)
3. Generate STAR-GO embeddings using S-BERT and customized Anc2vec
4. Generate ProtT5 residue embeddings for all proteins (**Note**: embedding files are large, ~150-200GB per file)
5. Prepare zero-shot evaluation data for DeepGOZero

**Available Options:**
- `--data-dir DIR`: Specify data directory (default: `datasets`)
- `--dataset TYPE`: Choose `pfresgo`, `deepgozero`, or `both` (default: `both`)
- `--device DEVICE`: Use `cuda`, `cpu`, or `mps` (default: `cuda`)
- `--skip-download`: Skip dataset download if already downloaded
- `--skip-go-embs`: Skip GO embedding generation
- `--skip-residue-embs`: Skip residue embedding generation
- `--skip-zero-shot`: Skip zero-shot data preparation

### Manual Setup

If you prefer to run steps individually:

1. Download datasets:
```bash
python bin/download_data.py --dataset both --data-dir datasets
```

2. Generate GO embeddings:
```bash
python bin/generate_go_embs.py --go-date 2020-06-01 --edition basic --device cuda
python bin/generate_go_embs.py --go-date 2021-11-16 --edition basic --device cuda
```

3. Copy embeddings to datasets:
```bash
cp embeddings/go-basic-2020-06-01.stargo.npy datasets/pfresgo/ontology.embeddings.npy
cp embeddings/go-basic-2020-06-01.sbert.npy datasets/pfresgo/ontology.sbert-embeddings.npy
cp embeddings/go-basic-2021-11-16.stargo.npy datasets/deepgozero/ontology.embeddings.npy
```

4. Generate residue embeddings:
```bash
python bin/generate_residue_embs.py --data-dir datasets/pfresgo --dataset-type pfresgo --ontology all --output-file datasets/pfresgo/per_residue_embeddings.h5
python bin/generate_residue_embs.py --data-dir datasets/deepgozero --dataset-type pugo --ontology all --output-file datasets/deepgozero/per_residue_embeddings.h5
```

5. Prepare zero-shot data (DeepGOZero only):
```bash
python bin/prepare_zero_shot_data.py --subontology all
```

## Datasets

This project supports two dataset formats with corresponding DataModule implementations:
- **PFresGODataModule**: For PFresGO datasets
- **PUGODataModule**: For PUGO and DeepGOZero datasets (which share the same format)

## GO Embeddings

GO term embeddings are generated using a two-stage approach combining SBERT and Anc2vec:
1. **SBERT (S-BioBert)**: Generates initial embeddings from GO term definitions
2. **Anc2vec**: Fine-tunes embeddings using the GO ontology structure

Our implementation uses a [modified anc2vec implementation](https://github.com/boun-tabi-lifelu/anc2vec) [(original anc2vec repository)](https://github.com/aedera/anc2vec). The `prepare_data.sh` script automatically generates these embeddings and places them in the appropriate directories.

## Model Training

All STAR-GO model variants can be trained using the `bin/train.py` script. For reference implementations, see `scripts/train.sh`, which contains example training commands for some variants.

By default, each subontology is trained separately. However, you can chain multiple training runs by specifying subontologies as a comma-separated list. For example, the following command trains all STAR-GO variants sequentially for the three GO subontologies (Biological Process, Molecular Function, and Cellular Component):

```bash
python ./bin/train.py -c ./configs/ordered_encdec_medium.toml -d ./datasets/pfresgo -s bp,mf,cc -r 2020 -b 32
```

**Note**: This assumes the PFresGO dataset has been downloaded to `./datasets/pfresgo` which is how the data script sets it up. The batch size argument is automatically adjusted to maintain the same effective batch size. Biological Process requires a lot of VRAM, so we recommend using a smaller batch size for it (effective `bs` won't change).


## Model Evaluation

Two evaluators are provided corresponding to the dataset formats:
- **PFresGOEvaluator**: For PFresGO datasets
- **PUGOEvaluator**: For PUGO and DeepGOZero datasets (includes zero-shot metrics calculation)

To evaluate models, use the `bin/evaluate.py` script. Example commands can be found in `scripts/evaluate.sh`.