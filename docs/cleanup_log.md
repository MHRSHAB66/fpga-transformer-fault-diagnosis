# Cleanup Log — 2026-08-24

## Deleted

| Path | دلیل |
|------|------|
| `data/raw/mafaulda_synthetic/` | داده سنتتیک debug — در git history محفوظ |
| `data/sample/` | پوشه خالی، فقط inspect_data قدیمی استفاده میکرد |
| `artifacts/cache/` | 8GB کش پنجره‌ها — preprocessing قرار است روی FPGA انجام شود |
| `artifacts/runs/tuning_20260616_161108/` | نتایج tuning — به `artifacts/tuning/` منتقل شد |
| `artifacts/runs/tuning_hij_20260813/` | نتایج tuning H/I/J — به `artifacts/tuning/` منتقل شد |
| `artifacts/runs/tuning_skipped_kaggle_20260801/` | نتایج tuning — به `artifacts/tuning/` منتقل شد |
| `artifacts/checkpoints/` | checkpoints قدیمی — به `artifacts/runs/` منتقل شدن |
| `experiments/` | خالی |
| `fpga-fault-tuning/` | خالی |
| `notebooks/` | خالی |
| `hls_f_int10/` | ادغام شد با `hls/f1_int10/` |

---

## Reorganized

### `artifacts/runs/`
قبلاً run های tuning و اصلی همه کنار هم بودن. الان:
```
artifacts/runs/
├── f1_1layer/        ← مدل deployed روی FPGA (int10، 97.55%)
├── 2layer_baseline/  ← مدل T4 Kaggle (98.22%، مرجع)
└── mlp_baseline/     ← baseline ساده Stage 3
```

### `artifacts/tuning/`
همه tuning reports از `artifacts/runs/` و `kaggle/tuning/` یکجا:
```
artifacts/tuning/
├── arch_tuning_report.json
├── hij_final_report.json
└── skipped_results_merged.json
```

### `artifacts/quantized/int10/`
نتایج quantization از `kaggle/int10_shard/` و `kaggle/sweep_multibit/`:
```
artifacts/quantized/int10/
├── fulltest_combined.json
└── bit_sweep.json
```

### `artifacts/analysis/frequency/`
چهار پوشه تحلیل فرکانسی که پراکنده بودن یکجا شدن:
```
artifacts/analysis/frequency/
├── frequency_audit/
├── channel_analysis/
├── frequency_investigation/
└── spectrum_analysis/
```

### `hls/`
دو پوشه جداگانه `hls/` و `hls_f_int10/` ادغام شدن:
```
hls/
├── f1_int10/   ← مدل deployed (با vitis/ و golden/)
└── 2layer/     ← مدل 2-layer قدیمی (مرجع)
```

---

## تغییر در `src/config.py`
- `DATA_SAMPLE_DIR` حذف شد (پوشه‌اش پاک شده بود)
- `INCLUDE_TAIL_WINDOW` / `TAIL_WINDOW_MODE` حذف شدن — همیشه False بود؛ preprocessing قرار است روی FPGA انجام شود و tail window در آن context معنا ندارد. کد مربوطه از `prepare_dataset.py`، `make_dataloaders.py`، `experiment_config.py` پاک شد؛ `ablation_tail_window.py` به `archive/` منتقل شد.
- `USE_SMOTE_AUGMENTATION` / `USE_NOISE_INJECTION` / `NOISE_INJECTION_STD` حذف شدن — همیشه False/unused بودن؛ references فقط در `archive/` بودن.
- `RESAMPLE_ENABLED` / `LOWPASS_FILTER_ENABLED` / `WIENER_FILTER_ENABLED` / `LOW_ENERGY_FILTER_ENABLED` و تمام پارامترهای مرتبط حذف شدن — همه همیشه False بودن. توابع `apply_lowpass_filter`، `apply_wiener_filter`، `apply_resample_array`، `apply_low_energy_filter` از `prepare_dataset.py` و `preprocessed_cache.py` پاک شدن. `MAFAULDA_SAMPLE_RATE = 50000` نگه داشته شد چون ابزارهای تحلیلی استفاده میکنن.
