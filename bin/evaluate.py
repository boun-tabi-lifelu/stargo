# # Evaluate
#

import sys

sys.path.append(".")
import itertools
import json
import os
import pickle
from pathlib import Path
from typing import Literal, Union

import click
import numpy as np
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm
import networkx as nx

from config import from_toml
from data.datamodule import PFresGODataModule, PUGODataModule
from model import TrainingModel, get_model_cls
from pfresgo_eval import Method, load_test_prots, protein_centric_aupr_curves
from pugo_eval import PUGOEvaluator

go_graph_map = {
    "2020": "datasets/pfresgo/2020-go.obo",
    "2024": "datasets/pfresgo/go.obo",
    "2020-anc2vec": "datasets/pfresgo/2020-go.obo",
}

def load_model(
    config,
    go_release: str,
    model_type: str,
    subontology: str,
    use_wandb: bool = False,
    dataset_extra: str = ""
):
    subontology_short = "".join([word[0] for word in subontology.split("_")])

    config.train.subontology = subontology  # override subontology

    model_name = f"contempro-{subontology_short}-{go_release}-{model_type}{dataset_extra}"

    if use_wandb:
        import wandb

        run = wandb.init(project="contempro", name=model_name + "-eval", job_type="eval")
        wandb.config.update(config)
        wandb.config.update({"model_name": model_name})
        wandb.config.update({"subontology": subontology})
        artifact = run.use_artifact(f"{model_name}:best")
        os.makedirs("trained_models", exist_ok=True)
        path = artifact.download(root="trained_models")
        print(path)
        model = get_model_cls(config.model.name)(config.model)
        module = TrainingModel.load_from_checkpoint(
            "trained_models/" + artifact.files()[0].name,
            model=model,
            training_config=config.train,
        )
    else:
        module = TrainingModel.load_from_checkpoint(
            f"trained_models/{model_name}.ckpt",
            model=get_model_cls(config.model.name)(config.model),
            training_config=config.train,
        )

    return module, model_name

def inference(model, model_name, dm, use_wandb: bool = False):
    model.eval()
    with torch.no_grad():
        batches_list = []

        for batch in tqdm(dm.test_dataloader()):
            result = sigmoid(
                model(
                    {
                        "embeddings": batch["embeddings"].to(model.device   ),
                        "attention_mask": batch["attention_mask"].to(model.device),
                        "go_embeddings": batch["go_embeddings"].to(model.device),
                    }
                )
            )
            batches_list.append(result)

        result = torch.cat(batches_list, dim=0)

    torch.save(result, f"evaluation/{model_name}_test_preds.pt")
    if use_wandb:
        import wandb

        wandb.finish()

    return result

def prep_eval_file(predictions_file: str, dm: PFresGODataModule):
    # Prepares evaluation file used by the PFresGO eval code

    # Load predictions
    test_preds = torch.load(predictions_file, map_location="cpu")
    print(f"Predictions shape: {test_preds.shape}")
    print(f"Number of test samples: {len(dm.test_dataset)}")
    print(f"Number of GO terms: {len(dm.test_dataset.go_term_list)}")

    all_annots = []
    for batch in dm.test_dataloader():
        all_annots.append(batch["annotations"])

    all_annots = torch.cat(all_annots, dim=0)

    eval_data = {
        "Y_true": all_annots.cpu().numpy(),
        "Y_pred": test_preds.cpu().numpy(),
        "goterms": dm.test_dataset.go_term_list,
        "proteins": dm.test_dataset.protein_ids,
    }

    # Save evaluation data
    with open("eval_results.pckl", "wb") as f:
        pickle.dump(eval_data, f)


def calculate_metrics(
    predictions_file: str,
    model_name: str,
    dm: Union[PFresGODataModule, PUGODataModule],
    subontology_short: str,
    use_wandb: bool = False,
    go_release: str = "2024",
):
    """Calculate metrics using either PUGO or PFresGO evaluation based on dataset type"""
    if use_wandb:
        import wandb
        wandb.init(
            project="contempro", name=model_name + "-eval-metrics", job_type="eval"
        )

    if isinstance(dm, PUGODataModule):
        # PUGO evaluation
        test_preds = torch.load(predictions_file, map_location="cpu")

        # Get true annotations
        true_annotations = []
        for batch in dm.test_dataloader():
            batch_annots = batch["annotations"].cpu().numpy()
            for annots in batch_annots:
                # Convert binary vector to set of GO IDs
                go_indices = np.where(annots == 1)[0]
                go_terms = {dm.test_dataset.go_term_list[i] for i in go_indices}
                true_annotations.append(go_terms)

        # Initialize evaluator
        evaluator = PUGOEvaluator('datasets/pugo/go.obo')

        # Calculate metrics
        results = evaluator.calculate_metrics(
            predictions=test_preds,
            true_annotations=true_annotations,
            go_terms=dm.test_dataset.go_term_list,
            subontology=subontology_short.lower()
        )

    else:
        # PFresGO evaluation
        prep_eval_file(predictions_file, dm)

        # Create evaluation method
        method = Method("Contempro", "eval_results.pckl", subontology_short, go_graph_map[go_release])

        # Load test protein indices
        test_prots, seqid_mtrx = load_test_prots(
            "./datasets/pfresgo/nrPDB-GO_2019.06.18_test.csv"
        )
        prot_idx = np.where(seqid_mtrx[:, 4] == 1)[0]

        # Calculate metrics
        micro_aupr, macro_aupr, _ = method._function_centric_aupr(keep_pidx=prot_idx)
        auc = method.AUC(keep_pidx=prot_idx)
        fmax = method.fmax(keep_pidx=prot_idx)

        results = {
            "Micro AUPR": micro_aupr,
            "Macro AUPR": macro_aupr,
            "AUC": auc,
            "Fmax": fmax
        }

    # Print results
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")

    results["model"] = model_name
    results = {
        k: float(v) if not isinstance(v, str) else v for k, v in results.items()
    }

    # Save results
    with open(f"evaluation/{model_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    if use_wandb:
        wandb.log(results)
        wandb.finish()

    return results

def calculate_average_depth(predictions_file: str, model_name: str, dm: PFresGODataModule, subontology_short: str, go_release: str):
    """Calculate the average depth of predicted GO terms weighted by prediction confidence.

    Returns:
        dict: Contains metrics about prediction depths:
            - weighted_avg_depth: Average depth weighted by prediction confidence
            - threshold_avg_depth: Average depth of terms above threshold
            - max_depth: Maximum depth of any predicted term
            - depth_distribution: Distribution of predictions across depths
    """
    test_preds = torch.load(predictions_file, map_location="cpu")
    prep_eval_file(predictions_file, dm)

    method = Method("Contempro", "eval_results.pckl", subontology_short, go_graph_map[go_release])
    go_graph = method.go_graph

    # Get the best threshold from Fmax calculation
    fscores, _, _, thresholds = method._protein_centric_fmax()
    best_threshold = thresholds[np.argmax(fscores)]

    # Calculate depths for each GO term
    go2depth = {}
    ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}
    root = ont2root[subontology_short.lower()]

    for go_term in method.goterms:
        if go_term in go_graph:
            # Find all paths to root and take the longest one
            paths = nx.all_simple_paths(go_graph, go_term, root)
            try:
                max_path_length = max(len(path) - 1 for path in paths)  # -1 because path length includes the term itself
                go2depth[go_term] = max_path_length
            except ValueError:  # If no paths found
                go2depth[go_term] = 0

    # Calculate metrics
    go_term_list = dm.test_dataset.go_term_list
    depth_array = np.array([go2depth.get(term, 0) for term in go_term_list])

    # Weighted average depth using prediction confidences
    weighted_depths = []
    threshold_depths = []
    depth_distribution = {i: 0 for i in range(max(go2depth.values()) + 1)}

    for pred in test_preds:
        # Weighted average calculation
        pred_numpy = pred.numpy()
        weighted_avg = np.average(depth_array, weights=pred_numpy)
        weighted_depths.append(weighted_avg)

        # Thresholded calculation
        above_threshold = pred_numpy > best_threshold
        if above_threshold.any():
            threshold_avg = np.mean(depth_array[above_threshold])
            threshold_depths.append(threshold_avg)

            # Update depth distribution
            for d in depth_array[above_threshold]:
                depth_distribution[int(d)] += 1

    results = {
        "weighted_avg_depth": float(np.mean(weighted_depths)),
        "threshold_avg_depth": float(np.mean(threshold_depths)) if threshold_depths else 0.0,
        "max_depth": float(max(go2depth.values())),
        "depth_distribution": {str(k): int(v) for k, v in depth_distribution.items()}
    }

    return results

@click.command()
@click.option("--config-file", '-c', default="configs/ordered_encdec_medium.toml")
@click.option("--data-dir", '-d', default="datasets/pfresgo")
@click.option("--go-release", '-r', default="2024")
@click.option("--use-wandb", '-w', default=False, is_flag=True)
@click.option("--subontology", '-s', default="biological_process")
@click.option("--skip-existing", '-x', default=False, is_flag=True)
@click.option("--skip-inference", '-X', default=False, is_flag=True)
@click.option("--skip-pfresgo", '-m', default=False, is_flag=True)
@click.option("--batch-size", '-b', default=32)
def main(
    config_file: str,
    go_release: str,
    use_wandb: bool,
    subontology: str,
    data_dir: str,
    skip_existing: bool,
    skip_inference: bool,
    skip_pfresgo: bool,
    batch_size: int,
):
    """
    Evaluates trained models by running inference and calculating metrics.

    The script performs:
    1. Loading model checkpoints and configs
    2. Running inference on test data
    3. Calculating metrics like AUPR, AUC, Fmax

    Multiple evaluations can be batched by providing comma-separated values for:
    - config_file: Different model architectures/configs to evaluate
    - go_release: Different GO release versions to test against (2020, 2024, 2020-anc2vec)
    - subontology: Different GO subontologies (BP/MF/CC) to evaluate

    Example:
    python evaluate.py \\
      -c configs/model1.toml,configs/model2.toml \\
      -r 2023,2024 \\
      -s biological_process,molecular_function \\
      -d datasets/pfresgo-2020 \\
      -w
    """
    config_files = config_file.split(',')
    go_releases = go_release.split(',')
    subontologies = subontology.split(',')

    for config_file, go_release, subontology in itertools.product(config_files, go_releases, subontologies):
        if subontology in ["bp", "cc", "mf"]:
            subontology = "biological_process" if subontology == "bp" else "cellular_component" if subontology == "cc" else "molecular_function"

        config = from_toml(config_file)

        config.train.subontology = subontology

        model_type = config_file.split("/")[-1].split(".")[0].replace("_", "-")

        data_root_dir = Path(config.train.data_dir) if data_dir is None else Path(data_dir)

        if 'pfresgo' not in str(data_root_dir):
            dataset_extra = "-" + data_root_dir.name
        else:
            dataset_extra = ""


        if 'pugo' in str(data_root_dir) or 'netgo' in str(data_root_dir):
            dm = PUGODataModule(
            data_dir=data_root_dir,
            batch_size=config.train.batch_size,
            num_workers=config.train.dm_num_workers,
            ontology=config.train.subontology,
            go_release=config.train.go_release,
            order_go_terms=config.train.order_go_terms,
            train_go_embeddings=config.model.train_go_embeddings,
            )
        elif 'pfresgo' in str(data_root_dir):
            dm = PFresGODataModule(
            data_dir=data_root_dir,
            batch_size=config.train.batch_size,
            num_workers=config.train.dm_num_workers,
            ontology=config.train.subontology,
            go_release=config.train.go_release,
            order_go_terms=config.train.order_go_terms,
            train_go_embeddings=config.model.train_go_embeddings,
            )
        else:
            raise ValueError(f"Invalid data directory: {data_dir} - could not determine dataset type")
        dm.setup("test")

        # If training embeddings, set go_input_dim := vocab size
        if config.model.train_go_embeddings:
            # override the go_input_dim to be the number of GO terms if training embeddings, in this case it's not the dimension but the vocab size
            config.model.go_input_dim = len(dm.test_dataset.go_term_list)

        model, model_name = load_model(
            config, go_release, model_type, subontology, use_wandb, dataset_extra
        )

        if skip_existing and os.path.exists(f"evaluation/{model_name}_test_preds.pt"):
            print(f"Skipping {model_name} evaluation because it already exists")
            continue

        if not skip_inference:
            predictions = inference(model, model_name, dm, use_wandb)

        subontology_short = "".join([word[0] for word in subontology.split("_")])
        if not skip_pfresgo:
            results = calculate_metrics(
                f"evaluation/{model_name}_test_preds.pt",
                model_name,
                dm,
                subontology_short,
                use_wandb,
                go_release,
            )
        else:
            # try to load existing metrics from file
            try:
                results = json.load(open(f"evaluation/{model_name}_metrics.json", "r"))
            except Exception as e:
                print(f"No metrics found for {model_name}, skipping")
                results = {}

        results["model"] = model_name
        results = {
            k: float(v) if not isinstance(v, str) else v for k, v in results.items()
        }  # Convert numpy types to native Python types

        with open(f"evaluation/{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2)  # Added indent for better readability
        if use_wandb:
            import wandb
            wandb.log(results)
            wandb.finish()

        print(f"Metrics saved to {model_name}_metrics.json")

if __name__ == "__main__":
    main()