# Drug-Gene-Disease Association Completion

A PyTorch implementation for joint **gene completion** and **disease completion** using entity features, a training-triplet hypergraph, and an auxiliary relational graph. The default model combines both graph branches and is trained with BCE and QFAL losses.

## Installation

Requires Python 3.10 or later. Install a PyTorch build appropriate for your CPU or CUDA environment, then install the project dependencies:

```bash
git clone https://github.com/RTSAK8/drug.git
cd drug
python -m pip install -r requirements.txt
```

## Usage

```bash
# Default: joint training with independent testing, seed 42
python main.py

# Five-fold cross-validation
python main.py protocol=cross_validation

# Compare the full model, no-QFAL variant, and hypergraph+BCE baseline
python main.py -m ablation=full,no_qfal,base protocol=cross_validation

# Select device and data directory
python main.py device=cuda:0 data_dir=/path/to/dataset

# Inspect the configuration without starting training
python main.py --cfg job
```

Defaults and available variants are defined in [`conf/`](conf/). Training uses both tasks and selects the best checkpoint by mean validation filtered query-macro MRR.

Independent testing uses PharmGKB (`triplets.csv`) for training/validation and DrugBank for testing after removing overlapping triplets. Cross-validation uses only PharmGKB. Both protocols use random triplet splits over a fixed entity vocabulary; they are not cold-start evaluations.

## Results

Each experiment writes a separate directory under `results/`, containing checkpoints, predictions, metrics, configuration, and data/split metadata. Reported metrics include AUROC, AUPR, average precision, F1, raw/filtered MRR, Hits@K, and query-level Recall@K. Test F1 uses a threshold selected on validation data.

Use `metrics_long.csv` for per-fold results and `summary.csv` for aggregation across seeds. Plot a completed run with:

```bash
python draw.py results/RUN/metrics_long.csv --output-dir figures
```
