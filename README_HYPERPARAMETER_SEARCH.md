# Hyperparameter Search Scripts - Quick Guide

Three different scripts for different GPU setups. Choose based on your hardware.

## 🚀 Quick Decision Guide

| GPUs Available | Script to Use | Speed (24 trials) |
|----------------|---------------|-------------------|
| **1 GPU** | `single_gpu_hyperparameter_search.py` | ~48 hours |
| **2-8 GPUs** | `parallel_search.py` ⭐ **RECOMMENDED** | ~12 hours (4 GPUs) |
| **4+ GPUs (large models)** | `hyperparameter_search.py` | ~20 hours |

## Scripts Overview

### 1. `single_gpu_hyperparameter_search.py` - Single GPU
**Use when:** Only 1 GPU available

```bash
python single_gpu_hyperparameter_search.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 12 \
  --use_all_writers
```

**Pros:**
- ✅ Simple, no distributed complexity
- ✅ Works on any machine with 1 GPU
- ✅ Fully resumable

**Cons:**
- ❌ Slowest (one trial at a time)

---

### 2. `parallel_search.py` - Parallel Trials ⭐ **BEST FOR MOST CASES**
**Use when:** 2+ GPUs available

```bash
python parallel_search.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

**How it works:**
- Runs **one trial per GPU** simultaneously
- 4 GPUs = 4 trials at once
- Linear speedup: 4 GPUs = 4× faster

**Pros:**
- ✅ **Fastest for hyperparameter search**
- ✅ Linear speedup with GPUs (4 GPUs = 4× faster)
- ✅ Simple one-command launch
- ✅ Fully resumable
- ✅ Automatic trial coordination via Optuna

**Cons:**
- ❌ Requires multiple GPUs

**See:** [PARALLEL_SEARCH.md](PARALLEL_SEARCH.md) for details

---

### 3. `hyperparameter_search.py` - Multi-GPU DDP per Trial
**Use when:** Each trial needs multiple GPUs (very large models)

```bash
python hyperparameter_search.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 12 \
  --use_all_writers
```

**How it works:**
- Runs one trial at a time
- Each trial uses **all GPUs** with DistributedDataParallel
- Faster training per trial, same number of trials

**Pros:**
- ✅ Faster training per trial (if model is large)
- ✅ Good for huge models that don't fit on 1 GPU

**Cons:**
- ❌ Only one trial at a time (no parallelism across trials)
- ❌ More complex (DDP overhead)
- ❌ Not ideal for hyperparameter search

---

## Performance Comparison

**Scenario:** 24 trials, 4 GPUs available, 2 hours per trial on 1 GPU

| Script | Trials Running | GPU Usage | Total Time | Speedup |
|--------|---------------|-----------|------------|---------|
| Single GPU | 1 at a time | 1 GPU | **48 hours** | 1× |
| **Parallel** ⭐ | **4 at a time** | **4 GPUs** | **12 hours** | **4×** |
| Multi-GPU DDP | 1 at a time (on 4 GPUs) | 4 GPUs | ~20 hours | 2.4× |

**Winner:** `parallel_search.py` for hyperparameter optimization!

---

## Feature Comparison

| Feature | Single GPU | Parallel ⭐ | Multi-GPU DDP |
|---------|-----------|----------|---------------|
| **Speed** | 1× | **4× (with 4 GPUs)** | 2.4× |
| **GPU Efficiency** | 100% (1 GPU) | **100% (all GPUs)** | 100% (all GPUs) |
| **Trial Parallelism** | No | **Yes** | No |
| **Code Complexity** | Simple | Simple | Complex (DDP) |
| **Resumable** | ✅ | ✅ | ✅ |
| **Memory per Trial** | Full GPU | Full GPU | Split across GPUs |
| **Best For** | 1 GPU only | **Most cases** | Huge models |

---

## Resumability (All Scripts)

All three scripts are **fully resumable**. If interrupted:

```bash
# Just run the same command again
python <script>.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

The script will:
- ✅ Load existing Optuna study from SQLite database
- ✅ Skip completed trials
- ✅ Continue from where it stopped
- ✅ Run remaining trials only

**See:** [RESUMABILITY.md](RESUMABILITY.md) for details

---

## Which Should I Use?

### ⭐ **Recommended: `parallel_search.py`**

Use this if you have **2 or more GPUs**. It's the fastest and simplest for hyperparameter search.

```bash
python parallel_search.py \
  --data_root /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --n_trials 24 \
  --use_all_writers
```

### Use `single_gpu_hyperparameter_search.py` if:
- You only have **1 GPU**
- You're testing the script first

### Use `hyperparameter_search.py` if:
- Your model is **too large** for a single GPU
- You need DDP for memory reasons
- You're okay with slower hyperparameter search

---

## Example Commands

### Quick Test (1-2 hours)
```bash
# Single GPU - 4 trials
python single_gpu_hyperparameter_search.py \
  --data_root /path/to/data \
  --checkpoint_dir ./test_search \
  --n_trials 4 \
  --num_writers_subset 7
```

### Production Run (Parallel - 4 GPUs)
```bash
# Parallel - 24 trials across 4 GPUs
python parallel_search.py \
  --data_root /project/mamro/Mirath_extracted_lines \
  --checkpoint_dir /project/mamro/checkpoints/search_20251225 \
  --n_trials 24 \
  --use_all_writers
```

### Large-Scale Search (Parallel - 8 GPUs)
```bash
# Parallel - 48 trials across 8 GPUs
python parallel_search.py \
  --data_root /project/mamro/Mirath_extracted_lines \
  --checkpoint_dir /project/mamro/checkpoints/search_large \
  --n_trials 48 \
  --use_all_writers
```

---

## Monitoring Progress

All scripts save to the same checkpoint structure:

```bash
# Check completed trials
ls -d checkpoints/trial_*/

# Count completed trials
ls -d checkpoints/trial_* | wc -l

# Check Optuna database
sqlite3 checkpoints/optuna_study.db \
  "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

For parallel search, also check worker logs:
```bash
tail -f checkpoints/logs/worker_gpu0_*.log
```

---

## Documentation

- **[RESUMABILITY.md](RESUMABILITY.md)** - How resumability works (all scripts)
- **[PARALLEL_SEARCH.md](PARALLEL_SEARCH.md)** - Detailed parallel search guide
- **This file** - Quick comparison and decision guide

---

## Summary

**If you have multiple GPUs, use `parallel_search.py`** - it's the fastest way to run hyperparameter search.

**If you only have 1 GPU, use `single_gpu_hyperparameter_search.py`** - simple and effective.

**Only use `hyperparameter_search.py` if your model is too large for a single GPU.**

Happy optimizing! 🚀
