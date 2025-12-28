# Hyperparameter Search - Google Colab Version

**Run Arabic writer identification hyperparameter search on Google Colab with A100 GPU!**

## ✨ Features

- 🎯 **Optimized for Colab A100**: Single GPU, perfect for Colab free tier or Pro
- 💾 **Google Drive Integration**: Automatic save to Google Drive
- ⚡ **Quick Test Mode**: Verify setup in 10-20 minutes
- 🔄 **Fully Resumable**: Continue from where you stopped if Colab disconnects
- 🛡️ **All Safeguards Included**: NaN detection, gradient monitoring, batch validation

## 🚀 Quick Start

### **Step 1: Open Notebook in Colab**

1. Upload `hyperparameter_search_colab.ipynb` to Google Colab
2. Or open directly: [Open in Colab](https://colab.research.google.com/)

### **Step 2: Enable GPU**

```
Runtime → Change runtime type → Hardware accelerator → GPU
```

Recommended: **A100 GPU** (available in Colab Pro)

### **Step 3: Follow the Notebook**

The notebook guides you through:
1. ✅ Mount Google Drive
2. ✅ Download script from GitHub (automatically gets latest version!)
3. ✅ Install dependencies
4. ✅ Configure paths
5. ✅ Run quick test (10-20 min)
6. ✅ Run full search (24-48 hours)
7. ✅ View results

## 📁 Files

| File | Description |
|------|-------------|
| `hyperparameter_search_colab.ipynb` | Jupyter notebook for Colab |
| `run_hyperparameter_search_colab.py` | Python script (Colab optimized) |
| `README_COLAB.md` | This file |

## 🎯 Quick Test vs Full Search

### **Quick Test** (Recommended First!)
```bash
!python run_hyperparameter_search_colab.py \
  --data_root /content/drive/MyDrive/Mirath_extracted_lines \
  --checkpoint_dir /content/drive/MyDrive/checkpoints \
  --quick_test
```

**Parameters:**
- 3 writers (instead of 21)
- 50 lines/writer (instead of 300)
- Batch size 32 (instead of 128)
- 5 epochs (instead of 70)
- 2 trials (instead of 12)

**Time:** ~10-20 minutes
**Purpose:** Verify data paths, GPU, and code work correctly

### **Full Search**
```bash
!python run_hyperparameter_search_colab.py \
  --data_root /content/drive/MyDrive/Mirath_extracted_lines \
  --checkpoint_dir /content/drive/MyDrive/checkpoints \
  --n_trials 12 \
  --use_all_writers
```

**Parameters:**
- All 21 writers
- 300 lines/writer
- Batch size 128
- 70 epochs per trial
- 12 trials total

**Time:** ~24-48 hours
**Purpose:** Full hyperparameter optimization

## 📊 Google Drive Structure

After running, your Google Drive will have:

```
MyDrive/
├── Mirath_extracted_lines/     # Your input data
│   ├── train/
│   ├── val/
│   └── test/
│
└── checkpoints/                 # Results (auto-saved)
    ├── optuna_study.db         # Optuna database (for resumability)
    ├── trial_000/              # Trial 0 results
    ├── trial_001/              # Trial 1 results
    ├── ...
    ├── trial_011/              # Trial 11 results
    └── best_overall/           # Best model across all trials
        ├── model.pth
        ├── config.json
        ├── summary.txt
        └── plots/
            ├── training_curves.png
            ├── confusion_matrix.png
            └── per_writer_results.png
```

## 🔄 Resumability

**If Colab disconnects:**

1. Re-run the same cell
2. Script detects existing Optuna database
3. Continues from where it stopped
4. No trials are lost!

**How it works:**
- Optuna database (`optuna_study.db`) stores trial state
- Each trial saves checkpoints to Google Drive
- Script automatically resumes incomplete trials

## 💡 Tips for Colab

### **Free Tier vs Colab Pro**

| Feature | Free Tier | Colab Pro |
|---------|-----------|-----------|
| GPU | T4 (16GB) | A100 (40GB) |
| Max Session | ~12 hours | ~24 hours |
| Batch Size | 64-96 | 128 |
| Speed | Slower | 3x faster |

**Recommendation:** Use Colab Pro for full search

### **Avoid Disconnects**

1. **Keep tab open**: Colab disconnects if you close the tab
2. **Disable sleep**: Keep your computer awake
3. **Check every few hours**: Monitor progress
4. **Use Colab Pro**: Longer sessions

### **Monitor Progress**

**Option 1:** Check Google Drive
```
Your checkpoints folder → trial_xxx folders
```

**Option 2:** Read logs (if you added logging)
```python
!tail -f /content/drive/MyDrive/checkpoints/logs/training.log
```

**Option 3:** Check Optuna database
```python
import optuna
study = optuna.load_study(
    study_name="arabic_writer_id",
    storage="sqlite:////content/drive/MyDrive/checkpoints/optuna_study.db"
)
print(f"Completed trials: {len(study.trials)}")
print(f"Best value: {study.best_value:.4f}")
```

## 🚨 Troubleshooting

### **"No GPU available"**
**Solution:** Enable GPU in Runtime settings
```
Runtime → Change runtime type → Hardware accelerator → GPU
```

### **"Google Drive not mounted"**
**Solution:** Run the mount cell
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **"Data not found"**
**Solution:** Update `DATA_ROOT` path to match your Google Drive structure

### **"Out of memory"**
**Solution 1:** Use `--quick_test` (smaller batch size)
**Solution 2:** Request A100 GPU (Colab Pro)
**Solution 3:** Reduce batch size manually in the script

### **"Colab disconnected, lost progress"**
**Solution:** No worries! Just re-run the cell. Script will resume automatically.

### **"Trial taking too long"**
**Expected times:**
- Quick test: 10-20 min
- Full trial (70 epochs): 2-4 hours on A100
- Full search (12 trials): 24-48 hours

## 🎓 Understanding Results

### **Best Model Location**
```
checkpoints/best_overall/
```

### **Trial Comparison**
Check `optuna_study.db` to see all trials:
```python
import optuna
import pandas as pd

study = optuna.load_study(...)
df = study.trials_dataframe()
df.sort_values('value', ascending=False).head(10)
```

### **Key Metrics**
- `val_macro_top1`: Validation Top-1 accuracy (main metric)
- `val_macro_top5`: Validation Top-5 accuracy
- `val_macro_map`: Mean Average Precision

## 📥 Download Results

**Option 1: Direct Download**
```python
from google.colab import files
files.download('/content/drive/MyDrive/checkpoints/best_overall/model.pth')
```

**Option 2: Zip and Download**
```bash
!cd /content/drive/MyDrive/checkpoints && \
 zip -r results.zip best_overall/ trial_*/summary.txt
```

Then download `results.zip` from Google Drive.

## 🆚 Differences from Server Version

| Feature | Server Version | Colab Version |
|---------|---------------|---------------|
| GPU Mode | Auto (1 or multi-GPU) | Single GPU only |
| Storage | Local filesystem | Google Drive |
| Parallel Trials | Yes (multi-GPU) | No |
| Max Memory | Depends on server | 40GB (A100) |
| Setup | Manual | Guided notebook |

## 📚 Additional Resources

- **Optuna Documentation**: https://optuna.org/
- **Colab Documentation**: https://colab.research.google.com/
- **PyTorch Documentation**: https://pytorch.org/

## 🐛 Known Issues

1. **Colab Timeout:** Free tier disconnects after 12 hours. Use Colab Pro for longer runs.
2. **Drive Sync Delay:** Google Drive may take a few seconds to sync files. Be patient.
3. **T4 GPU Memory:** T4 (16GB) may struggle with batch size 128. Use `--quick_test` or reduce batch size.

## ✅ Verification Checklist

Before running full search:

- [ ] GPU is enabled (check with `!nvidia-smi`)
- [ ] Google Drive is mounted (check `/content/drive`)
- [ ] Data path exists (check `!ls $DATA_ROOT/train`)
- [ ] Quick test completed successfully
- [ ] Results saved to Google Drive

## 🎉 You're Ready!

Open the notebook and start your hyperparameter search on Colab!

**Questions?** Check the notebook cells for detailed instructions and examples.
