# Selective Test-Time Debiasing for CLIP via Reward Gating

Code release for **Selective Test-Time Debiasing for CLIP via Reward Gating**.

This repository implements **Reward-Gated Test-Time Adaptation (RG-TTA)**, a selective test-time debiasing framework for CLIP-style vision-language models. Instead of applying the same debiasing operation to every query, RG-TTA uses a reward gate to decide whether a query is bias-sensitive and applies an attribute-balancing reward only when it is needed.

## Overview

Vision-language models such as CLIP show strong zero-shot performance, but person-centric queries can produce skewed demographic distributions. Uniform debiasing can reduce bias on sensitive queries, but it may also distort semantically meaningful information on bias-insensitive queries.

RG-TTA addresses this fairness-utility trade-off through input-conditioned adaptation:

1. Compute CLIP alignment between a query and candidate images.
2. Estimate whether the query is entangled with protected attributes through a reward gate.
3. Use an alignment-only reward for bias-insensitive queries.
4. Add an attribute-balancing reward for bias-sensitive queries.
5. Perform episodic policy-gradient updates and reset the adapted parameters for the next query.

## Key Features

- **Reward-gated debiasing**: activates fairness regularization only for bias-sensitive queries.
- **Episodic test-time adaptation**: adapts per query and resets model parameters to reduce unintended drift.
- **CLIP reward modeling**: separates the trainable policy CLIP model from the frozen reward CLIP model.
- **Bias-subspace scoring**: computes attribute-balancing scores in a protected-attribute subspace.
- **Fairness evaluation**: supports MaxSkew and NDKL evaluation for demographic distributional bias.
- **Dataset support**: includes loaders for FairFace, UTKFace, and FACET-style fairness evaluation.

## Repository Structure

The public repository is organized with the source files at the repository root:

```text
.
├── RL/
│   └── tta_rl_loop.py          # Episodic policy-gradient TTA loop
├── eval/
│   ├── measure_bias.py         # MaxSkew and NDKL metrics
│   └── metrics_logger.py       # Result/config/metadata logging
├── models/
│   └── text_tta_model.py       # CLIP text-encoder TTA wrapper
├── rewards/
│   ├── reward_model.py         # CLIP reward and combined reward computation
│   ├── debias_score.py         # Attribute-balancing debias score
│   └── subspace_test.py        # Bias-subspace construction utilities
├── args_parser.py              # Experiment arguments
├── datasets.py                 # FairFace, UTKFace, and FACET dataset classes
├── face_dataset_loader.py      # Dataset loading and image embedding utilities
├── prompt_loader.py            # Prompt/concept CSV loader
├── run_tta_experiment.py       # Main experiment entry point
├── run.sh                      # Example shell runner
├── requirements.txt            # Python dependencies
└── README.md
```

## Hardware & Environment

Recommended environment:

| Component | Recommendation |
|---|---|
| OS | Linux |
| Python | 3.10+ |
| GPU | CUDA-capable NVIDIA GPU |
| Core libraries | PyTorch, OpenAI CLIP, NumPy, pandas, Pillow, tqdm |
| Optional logging | Weights & Biases |

A GPU is recommended because RG-TTA performs test-time updates and CLIP embedding computation.

## Installation

The commands below assume the cloned directory is named `rg_tta`, so it can be imported as a Python package.

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git rg_tta
cd rg_tta

conda create -n rg_tta python=3.10 -y
conda activate rg_tta

pip install -r requirements.txt
```

The OpenAI CLIP dependency is installed through `requirements.txt`.

## Dataset Preparation

Datasets are not redistributed with this repository. Please download each dataset directly from its official source and follow the corresponding license and usage terms.

Expected local organization:

```text
data/
├── fairface/
│   ├── labels/
│   │   ├── train/train_labels.csv
│   │   └── val/val_labels.csv
│   └── imgs/train_val/
├── utkface/
│   └── UTKFace/
│       ├── utk_annotation.csv
│       └── <image files>
├── FACET/
│   ├── annotations/annotations.csv
│   ├── imgs_1/
│   ├── imgs_2/
│   └── imgs_3/
└── prompt_templates.csv
```

The prompt CSV should contain at least a `concept` column. The code constructs text queries such as:

```text
This person is <concept>.
A <concept> person.
```

Example:

```csv
template,concept
This person is {},doctor
A {} person,nurse
This person is {},poor
A {} person,rich
```

## Usage

Because the source files are placed at the repository root and use package-style imports, run the module from the parent directory or set `PYTHONPATH` accordingly.

```bash
# From inside the repository root
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"

python -m rg_tta.run_tta_experiment --help
```

Example FairFace race debiasing run:

```bash
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"

python -m rg_tta.run_tta_experiment \
  --dataset fairface \
  --attribute race \
  --gallery_split test \
  --device cuda \
  --prompt_csv ./data/prompt_templates.csv \
  --output_dir ./outputs \
  --run_name fairface_race_rg_tta \
  --policy_clip_model_name ViT-B/16 \
  --reward_clip_model_name ViT-L/14 \
  --reward_mode clip_plus_debias \
  --debias_lambda 1.0 \
  --debias_score_base instance_popularity \
  --debias_score_trace none \
  --subspace_mode test \
  --subspace_top_r 30 \
  --tta_steps 30 \
  --sample_k 16 \
  --lr 1e-4 \
  --wandb_mode disabled
```

Example UTKFace gender run:

```bash
python -m rg_tta.run_tta_experiment \
  --dataset utkface \
  --attribute gender \
  --gallery_split test \
  --device cuda \
  --prompt_csv ./data/prompt_templates.csv \
  --output_dir ./outputs \
  --run_name utkface_gender_rg_tta \
  --policy_clip_model_name ViT-B/16 \
  --reward_clip_model_name ViT-L/14 \
  --reward_mode clip_plus_debias \
  --wandb_mode disabled
```

For a different local package name, replace `rg_tta` in the module command with that package name.

## Main Arguments

| Argument | Description |
|---|---|
| `--dataset` | Evaluation dataset: `fairface`, `utkface`, or `facet` |
| `--attribute` | Protected attribute: `race`, `gender`, `age`, or `all` |
| `--gallery_split` | Dataset split used as the retrieval gallery |
| `--policy_clip_model_name` | CLIP backbone adapted during TTA, e.g., `ViT-B/16` |
| `--reward_clip_model_name` | Frozen CLIP model used for reward computation, e.g., `ViT-L/14` |
| `--reward_mode` | `clip_only` or `clip_plus_debias` |
| `--debias_lambda` | Weight for the attribute-balancing term |
| `--debias_score_base` | Debias score formulation |
| `--subspace_top_r` | Number of retrieved images per class for bias-subspace construction |
| `--tta_steps` | Number of policy-gradient update steps per query |
| `--sample_k` | Top-K candidates used in the reward and policy-gradient step |
| `--lr` | Learning rate for test-time adaptation |
| `--wandb_mode` | `disabled`, `offline`, or `online` |

## Outputs

Each run writes results under:

```text
outputs/<run_name>/
├── scores_summary.json     # Per-step average MaxSkew and NDKL
├── meta.json               # Prompt, dataset, and run metadata
└── config.json             # Reproducibility configuration
```

## Evaluation Metrics

This repository reports retrieval-based fairness metrics:

- **MaxSkew@k**: measures the maximum over-representation of a protected group in the top-k retrieved samples.
- **NDKL@k**: measures divergence between the retrieved protected-attribute distribution and the target distribution.

Lower MaxSkew and NDKL indicate fairer retrieval behavior.

## Experimental Results

The paper reports that RG-TTA reduces demographic skew while preserving or improving zero-shot utility. Representative ViT-B/16 results include:

| Setting | Original CLIP | RG-TTA |
|---|---:|---:|
| UTKFace gender MaxSkew | 0.114 | 0.051 |
| FACET gender MaxSkew | 0.478 | 0.053 |
| ImageNet Top-1 accuracy | 68.31 | 70.38 |
| ABLE | 77.39 | 80.87 |

See the paper for the full results across FairFace, UTKFace, FACET, ImageNet, and Flickr retrieval tasks.

## Notes on Responsible Use

This code is intended for research on fairness and bias mitigation in vision-language models. When using datasets with face images or protected-attribute annotations, follow all dataset licenses, consent restrictions, and institutional review requirements. Real-world deployment should include domain-specific auditing of both the reward model and attribute estimators.

## Citation

Please cite the paper if this repository is useful for your research.

```bibtex
@misc{rg_tta,
  title = {Selective Test-Time Debiasing for CLIP via Reward Gating},
  year = {2026}
}
```

## License

This project is released under the MIT License.
