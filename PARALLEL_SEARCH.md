# Parallel Hyperparameter Search

Run **multiple trials in parallel** across multiple GPUs for faster hyperparameter optimization.

## Overview

Instead of running one trial at a time (even with multi-GPU training), this approach runs **one trial per GPU simultaneously**, dramatically speeding up the search.

### Performance Comparison

**Single GPU (Sequential):**
- 12 trials × 2 hours/trial = **24 hours total**

**4 GPUs (Parallel with this script):**
- 12 trials ÷ 4 GPUs = 3 trials/GPU
- 3 trials × 2 hours/trial = **6 hours total**
- **4x faster!**

## How It Works

```
GPU 0: Trial #0 → Trial #4 → Trial #8  → Done
GPU 1: Trial #1 → Trial #5 → Trial #9  → Done
GPU 2: Trial #2 → Trial #6 → Trial #10 → Done
GPU 3: Trial #3 → Trial #7 → Trial #11 → Done
```

All workers:
✅ Share the same Optuna SQLite database
✅ Pull trials automatically (no conflicts)
✅ Write to same checkpoint directory
✅ Coordinate through Optuna's built-in locking

## Usage

### Basic Usage

```bash
python parallel_search.py \
  --data_root /path/to/Mirath_extracted_lines \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

### With 4 GPUs and 24 Trials

```bash
python parallel_search.py \
  --data_root /project/mamro/Mirath_extracted_lines \
  --checkpoint_dir /project/mamro/checkpoints/parallel_search_001 \
  --n_trials 24 \
  --use_all_writers
```

**Output:**
```
======================================================================
PARALLEL HYPERPARAMETER SEARCH LAUNCHER
======================================================================
Available GPUs: 4
Total trials: 24
Trials per GPU: ~6
Checkpoint dir: /project/mamro/checkpoints/parallel_search_001
Data root: /project/mamro/Mirath_extracted_lines
======================================================================

🚀 Launching workers...

  ✅ GPU 0: Worker launched (PID 12345)
     Log: checkpoints/logs/worker_gpu0_20251225_143022.log
  ✅ GPU 1: Worker launched (PID 12346)
     Log: checkpoints/logs/worker_gpu1_20251225_143024.log
  ✅ GPU 2: Worker launched (PID 12347)
     Log: checkpoints/logs/worker_gpu2_20251225_143026.log
  ✅ GPU 3: Worker launched (PID 12348)
     Log: checkpoints/logs/worker_gpu3_20251225_143028.log

======================================================================
ALL WORKERS LAUNCHED
======================================================================

📊 Monitor progress:
  • Check logs: checkpoints/logs
  • Check trials: checkpoints/trial_*
  • Database: checkpoints/optuna_study.db

⏳ Waiting for workers to complete...
```

## Monitoring Progress

### Check Worker Logs

Each GPU has its own log file:
```bash
# Real-time monitoring
tail -f checkpoints/logs/worker_gpu0_*.log

# Check all workers
ls -lh checkpoints/logs/
```

### Check Trial Progress

```bash
# See completed trials
ls checkpoints/trial_*/

# Count completed trials
ls -d checkpoints/trial_* | wc -l

# Check Optuna database
sqlite3 checkpoints/optuna_study.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"
```

### Monitor GPU Usage

```bash
# While search is running
watch -n 1 nvidia-smi
```

You should see all GPUs at high utilization.

## Interruption & Resume

### Interrupt Gracefully

Press **Ctrl+C** once:
```
🛑 INTERRUPT RECEIVED - Gracefully stopping all workers...
  Stopping worker on GPU 0...
  Stopping worker on GPU 1...
  Stopping worker on GPU 2...
  Stopping worker on GPU 3...
✅ All workers stopped
```

### Resume

Simply **run the same command again**:
```bash
python parallel_search.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

The script will:
- ✅ Load existing Optuna study from database
- ✅ Skip completed trials
- ✅ Run only remaining trials
- ✅ Continue until all 24 trials complete

## Directory Structure

```
checkpoint_dir/
├── optuna_study.db              # Shared Optuna database
├── logs/
│   ├── worker_gpu0_*.log        # GPU 0 worker logs
│   ├── worker_gpu1_*.log        # GPU 1 worker logs
│   ├── worker_gpu2_*.log        # GPU 2 worker logs
│   └── worker_gpu3_*.log        # GPU 3 worker logs
├── trial_000/                   # Completed trials
│   ├── config.json
│   ├── best_model.pth
│   └── ...
├── trial_001/
├── trial_002/
├── ...
├── trial_023/
├── summary/                     # Generated after all trials
│   ├── all_trials_comparison.png
│   ├── hyperparameter_relationships.png
│   └── top5_trials.csv
└── best_overall/                # Best model across all trials
    ├── best_model.pth
    └── config.json
```

## Advanced Options

### Limit Number of Writers

```bash
python parallel_search.py \
  --data_root /path/to/data \
  --checkpoint_dir ./checkpoints \
  --n_trials 24 \
  --num_writers_subset 7
```

### Adjust Trial Count

For optimal GPU utilization, use **multiples of GPU count**:
- 4 GPUs: 12, 16, 20, 24, 28, 32 trials
- 8 GPUs: 16, 24, 32, 40, 48 trials

```bash
# 32 trials on 4 GPUs = 8 trials per GPU
python parallel_search.py \
  --data_root /path/to/data \
  --checkpoint_dir ./checkpoints \
  --n_trials 32 \
  --use_all_writers
```

## Comparison with Other Approaches

### 1. Single GPU Sequential (single_gpu_hyperparameter_search.py)
```
Trial 0 → Trial 1 → Trial 2 → ... → Trial 11
Time: 12 × 2h = 24 hours
```
**Use when:** Only 1 GPU available

### 2. Multi-GPU DDP per Trial (hyperparameter_search.py)
```
Trial 0 (using 4 GPUs) → Trial 1 (using 4 GPUs) → ...
Time: 12 × 0.8h = 9.6 hours (faster training, same # trials)
```
**Use when:** Each trial is very slow and needs multiple GPUs

### 3. Parallel Trials (parallel_search.py) ⭐ **BEST FOR HYPERPARAMETER SEARCH**
```
GPU 0: Trial 0 → Trial 4 → Trial 8
GPU 1: Trial 1 → Trial 5 → Trial 9
GPU 2: Trial 2 → Trial 6 → Trial 10
GPU 3: Trial 3 → Trial 7 → Trial 11
Time: (12 ÷ 4) × 2h = 6 hours
```
**Use when:** Multiple GPUs available (RECOMMENDED)

## Troubleshooting

### Problem: "No GPUs available"

**Solution:** Check GPU visibility:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

### Problem: Workers fail to start

**Solution:** Check logs:
```bash
cat checkpoints/logs/worker_gpu*_*.log
```

Common issues:
- Data path not found
- Insufficient GPU memory
- Missing dependencies

### Problem: Database locked errors

**Solution:** This is normal! Optuna handles SQLite locking automatically. Workers will retry.

If persistent:
```bash
# Check for stale locks
lsof checkpoints/optuna_study.db

# If needed, wait for all workers to finish
pkill -f single_gpu_hyperparameter_search.py
```

### Problem: Uneven GPU utilization

**Solution:** This is expected! Trials finish at different times. Some GPUs may complete faster than others depending on:
- Hyperparameter combinations (some train faster)
- Early stopping (some trials pruned early)
- GPU variance

## Best Practices

### 1. Use Tmux/Screen for Long Runs

```bash
# Start tmux session
tmux new -s hyperparam_search

# Run search
python parallel_search.py --data_root ... --checkpoint_dir ... --n_trials 48

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t hyperparam_search
```

### 2. Set Resource Limits

If you have limited GPU memory:
```python
# Edit single_gpu_hyperparameter_search.py BaseConfig
BATCH_SIZE = 64  # Reduce if OOM
```

### 3. Monitor System Resources

```bash
# Terminal 1: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 2: System monitoring
htop

# Terminal 3: Worker logs
tail -f checkpoints/logs/worker_gpu0_*.log
```

### 4. Regular Checkpoints

The search saves automatically, but for extra safety:
```bash
# Backup checkpoint directory periodically
cp -r checkpoints checkpoints_backup_$(date +%Y%m%d_%H%M%S)
```

## FAQ

**Q: Can I run more workers than GPUs?**
A: Not recommended. Each worker uses 1 GPU fully. Running multiple workers per GPU causes memory issues and doesn't improve speed.

**Q: What if trials finish at different times?**
A: Normal! Fast trials (early stopped or simple hyperparams) finish quickly. Slow trials take longer. This is fine - GPUs will pull new trials as they become free.

**Q: Can I add more trials while running?**
A: Yes! Interrupt with Ctrl+C, then restart with higher `--n_trials`. Optuna will continue from where it left off.

**Q: Does this work on SLURM/HPC clusters?**
A: Yes! Just submit as a single job requesting multiple GPUs. The script handles GPU assignment automatically.

**Q: Can I prioritize certain GPUs?**
A: The script uses all available GPUs equally. To exclude GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,3 python parallel_search.py ...  # Skip GPU 2
```

## Performance Tips

1. **Trial Count:** Use multiples of GPU count for even distribution
2. **Batch Size:** Smaller batches = more trials fit in memory, but slower
3. **Early Stopping:** Patience=15 is good - stops bad trials early
4. **Pruning:** Optuna's MedianPruner kills bad trials automatically

## Summary

✅ **Faster:** N×speedup with N GPUs
✅ **Resumable:** Interrupt and continue anytime
✅ **Automatic:** Optuna coordinates everything
✅ **Simple:** One command to launch
✅ **Safe:** Graceful shutdown, no data loss

**Recommended for all multi-GPU hyperparameter searches!**
