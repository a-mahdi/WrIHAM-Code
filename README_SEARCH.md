# Hyperparameter Search - Auto Multi-GPU

**One script, automatic GPU detection, optimal performance.**

## Overview

`run_hyperparameter_search.py` automatically detects available GPUs and chooses the best execution mode:

- **1 GPU**: Sequential trials (one at a time)
- **2+ GPUs**: Parallel trials (one trial per GPU) for linear speedup

**No configuration needed** - just run the script!

## Quick Start

```bash
python run_hyperparameter_search.py \
  --data_root /path/to/Mirath_extracted_lines \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

That's it! The script automatically:
- ✅ Detects number of GPUs
- ✅ Chooses optimal mode (sequential or parallel)
- ✅ Distributes trials efficiently
- ✅ Coordinates via shared Optuna database
- ✅ Handles interruptions gracefully

## How It Works

### Single GPU (Automatic)

```
Detected: 1 GPU
Mode: Sequential

Trial 0 → Trial 1 → Trial 2 → ... → Trial 11
Time: 12 trials × 2h = 24 hours
```

### Multi-GPU (Automatic)

```
Detected: 4 GPUs
Mode: Parallel

GPU 0: Trial 0 → Trial 4 → Trial 8  → Done
GPU 1: Trial 1 → Trial 5 → Trial 9  → Done
GPU 2: Trial 2 → Trial 6 → Trial 10 → Done
GPU 3: Trial 3 → Trial 7 → Trial 11 → Done

Time: (12 ÷ 4) × 2h = 6 hours
Speedup: 4x faster!
```

## Performance

| GPUs | Mode | 24 Trials Time | Speedup |
|------|------|----------------|---------|
| 1 | Sequential | 48 hours | 1x |
| 2 | Parallel | 24 hours | 2x |
| 4 | Parallel | 12 hours | 4x |
| 8 | Parallel | 6 hours | 8x |

**Linear speedup with number of GPUs!**

## Features

### 🚀 Automatic Mode Selection
No need to choose scripts or configure anything. The script detects GPUs and optimizes automatically.

### 💾 Fully Resumable
Uses SQLite database to track progress. If interrupted:
```bash
# Just run the same command again
python run_hyperparameter_search.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

The script will:
- Load existing study
- Skip completed trials
- Continue from where it stopped

### 🛑 Graceful Shutdown
Press **Ctrl+C** to stop:
```
🛑 INTERRUPT RECEIVED - Gracefully stopping all workers...
  Stopping worker on GPU 0...
  Stopping worker on GPU 1...
  Stopping worker on GPU 2...
  Stopping worker on GPU 3...
✅ All workers stopped
```

All progress is saved automatically.

### 📊 Easy Monitoring

**Parallel mode creates individual logs per GPU:**
```bash
# Check worker logs
tail -f checkpoints/logs/worker_gpu0_*.log

# Monitor all GPUs
watch -n 1 nvidia-smi
```

**Check progress:**
```bash
# Count completed trials
ls -d checkpoints/trial_* | wc -l

# Check database
sqlite3 checkpoints/optuna_study.db \
  "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"
```

## Usage Examples

### Basic Usage (All Writers)
```bash
python run_hyperparameter_search.py \
  --data_root /path/to/Mirath_extracted_lines \
  --checkpoint_dir ./checkpoints \
  --n_trials 24 \
  --use_all_writers
```

### Subset of Writers
```bash
python run_hyperparameter_search.py \
  --data_root /path/to/data \
  --checkpoint_dir ./checkpoints \
  --n_trials 12 \
  --num_writers_subset 7
```

### Large-Scale Search
```bash
python run_hyperparameter_search.py \
  --data_root /path/to/data \
  --checkpoint_dir ./checkpoints_large \
  --n_trials 48 \
  --use_all_writers
```

## Output Structure

```
checkpoint_dir/
├── optuna_study.db              # Shared Optuna database (resumable)
├── logs/                         # Worker logs (parallel mode only)
│   ├── worker_gpu0_*.log
│   ├── worker_gpu1_*.log
│   └── ...
├── trial_000/                   # Individual trial results
│   ├── config.json
│   ├── best_model.pth
│   ├── metrics.json
│   └── plots/
├── trial_001/
├── ...
├── summary/                     # Created after all trials complete
│   ├── all_trials_comparison.png
│   ├── hyperparameter_relationships.png
│   └── top5_trials.csv
└── best_overall/                # Best model across all trials
    ├── best_model.pth
    ├── config.json
    └── final_report.txt
```

## Command Line Options

```
--data_root PATH           Path to Mirath_extracted_lines directory (required)
--checkpoint_dir PATH      Directory to save results (required)
--n_trials N              Total number of trials to run (default: 12)
--use_all_writers         Use all writers in the dataset
--num_writers_subset N    Number of writers to use if not all (default: 7)
```

## Technical Details

### Single GPU Mode
- Runs trials sequentially
- Full GPU utilization per trial
- Simple, straightforward execution

### Parallel Mode (Multi-GPU)
- Spawns one worker process per GPU
- Each worker restricted to single GPU via `CUDA_VISIBLE_DEVICES`
- Workers share Optuna SQLite database
- Automatic trial coordination (no conflicts)
- Workers run independently and may finish at different times

### Database Coordination
All modes use the same Optuna SQLite database at:
```
{checkpoint_dir}/optuna_study.db
```

This database:
- Tracks trial state (COMPLETE, PRUNED, FAILED, RUNNING)
- Stores hyperparameters and results
- Prevents duplicate trials
- Enables resumability
- Handles concurrent access automatically

## Troubleshooting

### No GPUs Detected
**Error:** `❌ No GPUs available. This script requires CUDA-capable GPUs.`

**Solution:**
```bash
# Check GPU visibility
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

### Worker Fails in Parallel Mode
**Check logs:**
```bash
cat checkpoints/logs/worker_gpu*_*.log
```

Common issues:
- Out of memory → Reduce batch size in BaseConfig
- Data path not found → Check --data_root
- Missing dependencies → Install requirements

### Database Locked
**Normal behavior!** Optuna handles SQLite locking automatically. Workers retry on lock.

If persistent:
```bash
# Wait for all workers to finish
# Or check for stale processes
ps aux | grep run_hyperparameter_search
```

## Best Practices

### 1. Use Tmux/Screen for Long Runs
```bash
tmux new -s hyperparam_search
python run_hyperparameter_search.py --data_root ... --checkpoint_dir ... --n_trials 48
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t hyperparam_search
```

### 2. Optimal Trial Count
Use multiples of GPU count for even distribution:
- 4 GPUs: 12, 16, 20, 24, 32, 40 trials
- 8 GPUs: 16, 24, 32, 40, 48, 64 trials

### 3. Monitor Resources
```bash
# Terminal 1: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 2: Worker logs (parallel mode)
tail -f checkpoints/logs/worker_gpu0_*.log
```

### 4. Regular Backups
```bash
# Backup checkpoint directory
cp -r checkpoints checkpoints_backup_$(date +%Y%m%d_%H%M%S)
```

## Comparison with Original Scripts

| Feature | Original Multi-GPU DDP | This Script (Auto) |
|---------|------------------------|-------------------|
| **Speed (4 GPUs)** | 2.4x (DDP per trial) | **4x (parallel trials)** |
| **Complexity** | High (DDP setup) | **Low (automatic)** |
| **Use Case** | Huge models | **Hyperparameter search** |
| **Setup** | Manual configuration | **Auto-detect** |
| **Resumable** | ✅ | ✅ |
| **Mode Selection** | Manual | **Automatic** |

**This script is the recommended approach for hyperparameter search.**

## Migration from Old Scripts

If you were using:
- `single_gpu_hyperparameter_search.py` → Use `run_hyperparameter_search.py` (same commands work)
- `parallel_search.py` → Use `run_hyperparameter_search.py` (same commands work)
- `hyperparameter_search.py` → Use `run_hyperparameter_search.py` (better for hyperparam search)

**Same command line arguments, automatic optimization!**

## FAQ

**Q: Will it work on my 8 GPU cluster?**
A: Yes! Automatically detects all 8 GPUs and runs 8 trials in parallel.

**Q: What if I only have 1 GPU?**
A: Works perfectly! Auto-detects and runs in sequential mode.

**Q: Can I limit it to use only 2 GPUs out of 4?**
A: Yes, use `CUDA_VISIBLE_DEVICES`:
```bash
CUDA_VISIBLE_DEVICES=0,1 python run_hyperparameter_search.py ...
```

**Q: Is it resumable?**
A: Yes! Just run the same command again. It loads from the SQLite database and continues.

**Q: Can I add more trials after finishing?**
A: Yes! Run with higher `--n_trials`:
```bash
# First run: 12 trials
python run_hyperparameter_search.py ... --n_trials 12

# Add 12 more (total 24)
python run_hyperparameter_search.py ... --n_trials 24
```

**Q: Does it work on SLURM/HPC clusters?**
A: Yes! Just request multiple GPUs in your SLURM job. The script handles the rest.

## Summary

✅ **One script** for all GPU configurations
✅ **Automatic** mode detection (single or parallel)
✅ **Linear speedup** with number of GPUs
✅ **Fully resumable** with SQLite database
✅ **Graceful shutdown** with Ctrl+C
✅ **Easy monitoring** with individual logs
✅ **Production-ready** for HPC clusters

**Just run it - the script optimizes everything automatically!** 🚀

## See Also

- [RESUMABILITY.md](RESUMABILITY.md) - Detailed resumability documentation
- Original multi-GPU DDP script: `hyperparameter_search.py` (still available for huge models)
