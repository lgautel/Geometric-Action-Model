# LIBERO-Plus 成绩复现（官方 HF checkpoint 评测）

本目录存放 **Geometric-Action-Model** 在 LIBERO-Plus 上复现论文数字（**85.5%** overall / **83.1%** camera）所需的脚本、日志与结果。默认路径是评测官方发布的 suite checkpoint，**不是**从零重训。

论文 / 代码：

- [arXiv:2606.17046](https://arxiv.org/abs/2606.17046)
- [项目页](https://cvlab-kaist.github.io/Geometric-Action-Model/)
- [HF checkpoints](https://huggingface.co/SeonghuJeon/3da-libero-gam)
- 仓库 [README](../../README.md) · [docs/evaluation.md](../../docs/evaluation.md)

**当前对照表：** [`results/PAPER_COMPARE.md`](results/PAPER_COMPARE.md)（与 [`results/STATUS.md`](results/STATUS.md) 同步生成）

## 复现进度（2026-07-12 完成）

四 suite 全量评测完成（10030 episodes）。对照表：[`results/PAPER_COMPARE.md`](results/PAPER_COMPARE.md)

| Suite | Episodes | SR | 状态 |
|-------|---------:|---:|------|
| spatial | 2203/2402 | **91.72%** | 完成 |
| object | 2202/2518 | **87.45%** | 完成 |
| goal | 1970/2591 | **76.03%** | 完成 |
| long | 1971/2519 | **78.25%** | 完成 |
| **Overall** | **8346/10030** | **83.21%** | vs 论文 85.5%（−2.29 pp） |
| **Camera** | **1259/1599** | **78.74%** | vs 论文 83.1%（−4.36 pp） |

协议：官方 HF suite ckpt，`--plus`，`qpos=original`，1 trial/task。

## 环境

| 项 | 值 |
|----|-----|
| Python venv | `/mnt/r/VENV/gam/`（editable 安装本仓库） |
| GPU | 8× NVIDIA H200 |
| 协议 | `--plus`，`qpos=original`，`num-trials-per-task=1`，全扰动 |

## 一键流程

```bash
cd /path/to/Geometric-Action-Model
source b/liberopls/env.sh

bash b/liberopls/scripts/00_bootstrap_venv.sh
bash b/liberopls/scripts/01_setup_sources.sh
bash b/liberopls/scripts/02_download_weights.sh

# 冒烟（可选：TASK_IDS=0,1 更快）
TASK_IDS=0,1 bash b/liberopls/scripts/03_smoke_eval.sh

# 全量四 suite（约 10030 episodes）
bash b/liberopls/scripts/04_run_libero_plus_all.sh

# 对照论文
python b/liberopls/scripts/05_aggregate_report.py
```

结果写入 `b/liberopls/results/`。

## 目录

```text
b/liberopls/
  env.sh
  README.md
  scripts/00_bootstrap_venv.sh … 06_optional_finetune.sh
  scripts/07_resume_goal_long.sh   # 续跑 goal+long
  scripts/05_aggregate_report.py
  results/          # suite summaries + PAPER_COMPARE.md
  logs/
```

## 已知问题与修复

1. **`flex_attention` batch-dim compile 失败**  
   Batched eval 变化 policy batch size 时，`torch.compile(flex_attention, dynamic=False)` 触发 inductor `Batch dimension must match`，几乎全失败。  
   - 源码：`future_predictor.py` / `da3_giant_encoder.py` 对 flex 使用 eager。  
   - 启动默认：`MAX_BATCH_SIZE=1`，`PARALLEL_ENVS_PER_GPU=8`。

2. **goal env-worker 初始化超时（900s）**  
   首轮 goal ~2030/2591 后 4 个 shard `TimeoutError: env worker timed out waiting for worker_init`，launcher `exit 1` 导致 **long 未启动**。  
   - `scripts/run_hf_gam_libero_plus_eval.sh` 现支持 `ENV_WORKER_TIMEOUT_SEC` / `ROLLOUT_WALL_TIMEOUT_SEC`（默认 1800）。  
   - 用 `07_resume_goal_long.sh` 归档残缺 goal 后重跑。

## 排障

- `opencv` 拉回带 GUI 的包：重新执行 `00` 里的 headless 强制安装。
- 缺 `track4world_da3.pth`：`02` 会从 `TencentARC/Track4World` 拉取。
- editable 安装后仍需 `source b/liberopls/env.sh`（LIBERO 走 PYTHONPATH）。
