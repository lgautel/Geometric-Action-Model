# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GAM (Geometric Action Model) — a language-conditioned robot manipulation policy that adapts the DA3-Giant geometric foundation model into a shared backbone for perception, future prediction, and action decoding. Evaluated on LIBERO and LIBERO-Plus benchmarks. The released model is 1.4B parameters.

## Build & Run Commands

**Environment setup** (conda or venv, then source deps):
```bash
conda env create -f environment.yml && conda activate gam-libero
# OR: python3.12 -m venv .venv && source .venv/bin/activate && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && pip install -r requirements.txt
bash scripts/setup_sources.sh
bash scripts/setup_libero_plus.sh --download-assets
```

**Required env vars** (every shell session):
```bash
export DA3_ROOT=/path/to/this_repo
export DA3_LIBERO_SOURCE_DIR=$DA3_ROOT/LIBERO
export DA3_LIBERO_PLUS_DIR=$DA3_ROOT/LIBERO-plus
export PYTHONPATH=$DA3_ROOT/src:$DA3_LIBERO_PLUS_DIR:$DA3_LIBERO_SOURCE_DIR:$PYTHONPATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

**Smoke test (single GPU, no data needed)** — predictor only:
```bash
PYTHONPATH=src:$PYTHONPATH python scripts/smoke_gam_predictor.py
```

**Smoke training run (single GPU, needs data)**:
```bash
PYTHONPATH=src:$PYTHONPATH python src/train_robot.py \
  --config configs/training/libero_unified/smoke/gam_chunk2.yaml \
  --single-gpu --set training.max_steps=1
```

**Multi-GPU training with DeepSpeed ZeRO-2**:
```bash
PYTHONPATH=src:$PYTHONPATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --set stage_1.ckpt_path=$GAM_PRETRAINED_CKPT --wandb
```

**LIBERO-Plus evaluation** (multi-GPU sharded):
```bash
GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh spatial
```

**LIBERO evaluation**:
```bash
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --suites libero_spatial --num-trials-per-task 5
```

## Architecture

### Pipeline: DA3 Backbone → GAM Predictor → Action Head

The model has three stages chained in a single forward pass:

1. **DA3GiantEncoder** (`src/robot/modeling/da3_giant_encoder.py`) — The Depth-Anything-3 Giant backbone, split at block 12. Blocks 0–12 ("shallow") produce per-view visual tokens (256 patches per view). Blocks 13–39 ("deep") run block-causal attention across timesteps. The backbone source lives in an external `Depth-Anything-3/` checkout added to sys.path at runtime.

2. **GAMFuturePredictor** (`src/robot/modeling/future_predictor.py`) — A 12-layer block-autoregressive transformer that takes DA3 block-12 visual tokens, proprio history, past actions, and CLIP language features. Uses RoPE (axial 4D for visual patches, 1D for proprio/action tokens), QK-RMSNorm, SwiGLU, and flex_attention with BlockMask for causal masking. Predicts future visual tokens, proprio, and action tokens. flex_attention must be torch.compile'd — eager calls OOM.

3. **ActionHeadV2** (`src/robot/modeling/action_head_v2.py`) — MLP-ResNet head that decodes action tokens into continuous 7-DoF delta actions (6DoF + gripper). Supports chunked prediction (multiple sub-actions per token) with learned chunk position encoding. Mean-pools across camera views.

### Training Path

`src/train_robot.py` is the training entrypoint. It wraps the three components in a `DA3FineTuneModel` and runs the unified forward+loss computation from `src/robot/losses/unified_loss.py`. The loss involves:
- Student DA3 on H past frames (shallow encode + predictor + deep propagation)
- Frozen teacher DA3 on all T frames (future L2 feature targets)
- Action L1 loss, future feature distillation loss, optional depth decode loss, optional SIGReg anti-collapse regularization

Training uses OmegaConf YAML configs with `--set key=value` overrides. Configs live under `configs/training/libero_unified/`.

Helper modules are factored into `src/gam/training/` (checkpoint, data, distributed, ema, metrics, model, optim) — these are imported into `train_robot.py` but the main training loop stays in that file.

### Evaluation Path

`src/eval_libero_unified.py` is the standalone rollout evaluation entrypoint. It rebuilds the model from a training config + checkpoint, then runs closed-loop rollouts in LIBERO/LIBERO-Plus simulators. Multi-GPU eval is handled by the shell launcher `scripts/run_hf_gam_libero_plus_eval.sh` which spawns one process per GPU with `--shard-index/--shard-count` and aggregates results.

LIBERO-Plus evaluation helpers are in `src/gam/evaluation/` (libero_plus.py for perturbation classification, registry.py for result tracking, video.py for rollout video generation).

### Data

`src/robot/data/dataset.py` provides `Dataset` classes that read LIBERO HDF5 demo files with embedded depth. Action and proprio normalization stats are loaded from `data/libero_noop/_stats/`. Camera images are rotated 180° for the DA3 input convention (`da3_input_rotate180: true` in config).

### Conditioning

- **Language**: CLIP ViT-L/14 text features (77 tokens, 768-dim) fed to the predictor
- **Proprioception**: 7-dim joint state, conditioned via `ProprioConditioner`
- **Text**: Task descriptions via `TextConditioner`

Both conditioners are in `src/robot/modeling/conditioning.py`.

### Key Config Keys

| Key | Controls |
|-----|----------|
| `da3_finetune.freeze_blocks_before` | DA3 block freeze boundary (13 = blocks 0–12 frozen as geometric encoder) |
| `da3_finetune.n_action_steps` | Action steps per GAM token sequence |
| `action_head.chunk_size` | Low-level actions per action-head token |
| `predictor.H_choices` / `H_weights` | History length sampling during training |
| `predictor.lambda_feat_future` | Future feature distillation weight |
| `training.compile` | torch.compile around training model |

### CUDA Graph Inference

Set `DA3_MAX_OPTIMIZE=1 DA3_COMPILE_INFERENCE_MODE=reduce-overhead DA3_FUSE_SHALLOW=1 DA3_SKIP_FULL_ENCODE=1` to fuse the entire h=1 inference path into a single CUDA graph for latency measurement (6.9ms reported).

## External Dependencies (Source-Installed)

- **Depth-Anything-3**: DA3 backbone, cloned to `./Depth-Anything-3/`, used via sys.path (not pip-installed)
- **LIBERO**: Benchmark suite, cloned to `./LIBERO/`, used via PYTHONPATH
- **LIBERO-Plus**: Perturbed evaluation variant, cloned to `./LIBERO-plus/`

All three are set up by `scripts/setup_sources.sh`. LIBERO's own requirements are intentionally ignored — its runtime deps (bddl, easydict, future) are pinned in this repo's `requirements.txt`.

## Important Constraints

- Python 3.10+, torch >= 2.5 required (flex_attention / BlockMask)
- `PYTHONPATH` must include `src/`, LIBERO, and LIBERO-Plus roots — the project is not pip-installable for import purposes
- Headless rendering requires system GL packages: `libgl1 libglvnd0 libegl1 libgles2 libosmesa6 libglfw3`
- opencv-python-headless must be used instead of opencv-python on headless servers (robosuite pulls the wrong one)
