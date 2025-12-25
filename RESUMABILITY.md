# Hyperparameter Search - Resumability Features

## Overview

The hyperparameter search script (`hyperparameter_search.py`) now includes **full resumability**. If the process stops for any reason (crash, power failure, manual interruption, etc.), it will automatically resume from where it left off.

## Key Features

### 1. Persistent Study Storage
- **SQLite Database**: Study state is saved to `{checkpoint_dir}/optuna_study.db`
- **Automatic Persistence**: Every trial completion is saved to disk immediately
- **No Data Loss**: Even if the process crashes mid-trial, completed trials are preserved

### 2. Automatic Resume Detection
When you restart the script with the same `--checkpoint_dir`:
- ✅ Automatically detects existing study database
- ✅ Loads all completed trials
- ✅ Continues from the next trial number
- ✅ Shows resume statistics:
  - Number of completed trials
  - Number of pruned trials
  - Number of failed trials
  - Best value so far
  - Remaining trials to run

### 3. Robust Error Handling
- **Trial Failures**: If a single trial fails, it's marked as FAILED and logged
- **Error Logs**: Each failed trial gets an `error.log` file in its trial directory
- **Continuation**: Study continues with next trial instead of crashing
- **Pruned Trials**: Properly handled by Optuna's pruning mechanism

## Usage

### Starting a New Search
```bash
python hyperparameter_search.py \
  --data_root /path/to/Mirath_extracted_lines \
  --checkpoint_dir /path/to/checkpoints/search_001 \
  --n_trials 12 \
  --use_all_writers
```

**Output:**
```
======================================================================
STARTING NEW HYPERPARAMETER SEARCH
======================================================================
Study database: /path/to/checkpoints/search_001/optuna_study.db
Number of trials: 12
Optimization metric: Validation Macro Top-1 Accuracy
Note: This search is resumable. You can safely stop and restart.
======================================================================
```

### Resuming After Interruption
Simply run the **exact same command** again:

```bash
python hyperparameter_search.py \
  --data_root /path/to/Mirath_extracted_lines \
  --checkpoint_dir /path/to/checkpoints/search_001 \
  --n_trials 12 \
  --use_all_writers
```

**Output:**
```
======================================================================
RESUMING EXISTING STUDY
======================================================================
Study loaded from: /path/to/checkpoints/search_001/optuna_study.db
Completed trials: 5
Pruned trials: 1
Failed trials: 0
Remaining trials: 6
Best value so far: 87.45%
======================================================================
```

The script will:
1. Load the existing study
2. Skip the 5 completed trials
3. Continue from trial #6
4. Run the remaining 6 trials

### Adding More Trials
To run additional trials beyond the original number:

```bash
# Original search had 12 trials, all completed
# Run 8 more trials (total will be 20)
python hyperparameter_search.py \
  --data_root /path/to/Mirath_extracted_lines \
  --checkpoint_dir /path/to/checkpoints/search_001 \
  --n_trials 20 \
  --use_all_writers
```

**Output:**
```
======================================================================
RESUMING EXISTING STUDY
======================================================================
Study loaded from: /path/to/checkpoints/search_001/optuna_study.db
Completed trials: 12
Pruned trials: 0
Failed trials: 0
Remaining trials: 8
Best value so far: 89.23%
======================================================================
```

## Directory Structure

After a crash and resume, the checkpoint directory will look like:

```
checkpoint_dir/
├── optuna_study.db              # ← Persistent study database
├── trial_000/                   # ← Completed before crash
│   ├── config.json
│   ├── best_model.pth
│   ├── final_model.pth
│   └── ...
├── trial_001/                   # ← Completed before crash
│   └── ...
├── trial_002/                   # ← Completed before crash
│   └── ...
├── trial_003/                   # ← Crashed mid-training (partial)
│   ├── config.json
│   ├── latest_checkpoint.pth   # ← May exist
│   └── error.log               # ← If error occurred
├── trial_004/                   # ← Created after resume
│   └── ...
└── summary/
    └── ...
```

## What Happens to Partial Trials?

If a trial is **interrupted mid-training**:
- ✅ It's marked as FAILED in the Optuna database
- ✅ An error log is saved (if possible)
- ✅ The trial directory may contain partial checkpoints
- ✅ On resume, a **new trial** will be started (different hyperparameters)
- ✅ The partial trial does **not** resume (Optuna will suggest new hyperparameters)

## Safety Features

1. **Idempotent Restarts**: Running the same command multiple times is safe
2. **No Double Work**: Completed trials are never re-run
3. **Atomic Saves**: Trial results are committed to database atomically
4. **Error Isolation**: One failed trial doesn't affect others

## Best Practices

### 1. Use Unique Checkpoint Directories
```bash
# Good - different experiments have different directories
python hyperparameter_search.py --checkpoint_dir ./exp_001 --n_trials 12
python hyperparameter_search.py --checkpoint_dir ./exp_002 --n_trials 12
```

### 2. Keep Consistent Parameters
When resuming, use the **same** `--n_trials` value unless you want to add more trials:
```bash
# First run (interrupted at trial 3)
python hyperparameter_search.py --checkpoint_dir ./exp --n_trials 12

# Resume (will complete 12 trials total)
python hyperparameter_search.py --checkpoint_dir ./exp --n_trials 12
```

### 3. Monitor the Database
Check study progress:
```bash
sqlite3 checkpoint_dir/optuna_study.db
sqlite> SELECT trial_id, state, value FROM trials;
```

### 4. Backup Important Studies
```bash
# Backup the entire checkpoint directory
cp -r checkpoint_dir/ checkpoint_dir_backup/
```

## Limitations

1. **Per-GPU Memory**: If a trial OOMs, it will fail and not resume
   - Solution: Reduce batch size or use fewer GPUs

2. **Hyperparameter Space Changes**: If you modify the hyperparameter search space, old trials may have different parameter sets
   - Solution: Start a new study with a fresh checkpoint directory

3. **Multi-GPU Coordination**: If multi-GPU training crashes, all GPUs need to restart together
   - This is handled automatically by the script

## Troubleshooting

### Problem: "No such table: trials" error
**Cause**: Corrupted database file

**Solution**:
```bash
# Delete corrupted database (will lose progress)
rm checkpoint_dir/optuna_study.db
# Restart the search
python hyperparameter_search.py ...
```

### Problem: Script keeps failing at the same trial
**Cause**: Specific hyperparameter combination causes OOM or other error

**Solution**: Optuna will mark it as failed and try different hyperparameters automatically. Check `error.log` in the trial directory for details.

### Problem: Want to start fresh but keep old results
**Solution**:
```bash
# Archive old study
mv checkpoint_dir checkpoint_dir_archived_$(date +%Y%m%d)
# Start new study
python hyperparameter_search.py --checkpoint_dir checkpoint_dir ...
```

## Technical Implementation

The resumability is implemented through:

1. **SQLite Storage Backend**:
   ```python
   storage = f'sqlite:///{checkpoint_dir}/optuna_study.db'
   study = optuna.load_study(study_name='...', storage=storage)
   ```

2. **Trial State Tracking**: Optuna tracks each trial's state:
   - `COMPLETE`: Successfully finished
   - `PRUNED`: Early stopped by pruning
   - `FAIL`: Crashed or errored
   - `RUNNING`: Currently executing

3. **Remaining Trials Calculation**:
   ```python
   n_completed = len([t for t in study.trials if t.state == COMPLETE])
   n_remaining = max(0, n_trials - n_completed)
   study.optimize(..., n_trials=n_remaining)
   ```

4. **Exception Handling**:
   - Pruned trials: Re-raised to Optuna
   - Other exceptions: Logged and re-raised to mark as FAILED

## Summary

✅ **Fully Resumable**: Stop and restart anytime
✅ **Zero Data Loss**: All completed trials preserved
✅ **Error Tolerant**: Failed trials don't crash the entire search
✅ **Easy to Use**: Just re-run the same command
✅ **Production Ready**: Safe for long-running experiments on HPC clusters
