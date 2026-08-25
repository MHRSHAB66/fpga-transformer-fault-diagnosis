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
- `INCLUDE_TAIL_WINDOW` / `TAIL_WINDOW_MODE` حذف شدن — همیشه False بود؛ آزمایش‌ها نشان داد پنجره‌های ناقص (کمتر از ۱۰۲۴ نمونه) نویز اضافه می‌کنند و دقت را بهبود نمی‌دهند، پس تصمیم گرفتیم اصلاً tail window نداشته باشیم. کد مربوطه از `prepare_dataset.py`، `make_dataloaders.py`، `experiment_config.py` پاک شد؛ `ablation_tail_window.py` به `archive/` منتقل شد.
- `USE_SMOTE_AUGMENTATION` / `USE_NOISE_INJECTION` / `NOISE_INJECTION_STD` حذف شدن — همیشه False/unused بودن؛ references فقط در `archive/` بودن.
- `RESAMPLE_ENABLED` / `LOWPASS_FILTER_ENABLED` / `WIENER_FILTER_ENABLED` / `LOW_ENERGY_FILTER_ENABLED` و تمام پارامترهای مرتبط حذف شدن — همه همیشه False بودن. توابع `apply_lowpass_filter`، `apply_wiener_filter`، `apply_resample_array`، `apply_low_energy_filter` از `prepare_dataset.py` و `preprocessed_cache.py` پاک شدن. `MAFAULDA_SAMPLE_RATE = 50000` نگه داشته شد چون ابزارهای تحلیلی استفاده میکنن.
- `CHANNEL_SELECTION` / `_VIBRATION_CHANNEL_INDICES` / `NUM_ACTIVE_CHANNELS` و if/elif block مربوطه حذف شدن — فقط "all" پیاده‌سازی شده بود. `INPUT_CHANNELS = EXPECTED_NUM_COLUMNS` مستقیم set شد.
- sanity check مربوط به `NORMALIZATION_SCOPE` حذف شد — constant رو با خودش چک میکرد.
- `ACTIVE_CHANNEL_INDICES` حذف شد (2026-08-25) — در pipeline اصلی استفاده نمی‌شد؛ فقط یک comment در `src/tools/channel_analysis.py` به آن اشاره داشت. `INPUT_CHANNELS = 8` نگه داشته شد چون `transformer_baseline.py`، `evaluate_model.py`، و `experiment_config.py` از آن import می‌کنند.

## آرشیو `src/tools/` به `archive/tools/`

| فایل | دلیل |
|------|------|
| `test_tail_window.py` | تست tail window که از config حذف شد |
| `signal_preprocessing.py` | مربوط به filter flags که حذف شدن |
| `run_skipped_tuning_configs.py` | tuning تمام شده |
| `inspect_data.py` | از `DATA_SAMPLE_DIR` استفاده میکرد که حذف شد |
| `build_deck.py` / `build_native_deck.py` / `build_presentation.py` | ابزارهای presentation قدیمی |
| `build_figures.py` / `deck_figures.py` | ابزارهای figure قدیمی |
| `add_notes.py` / `update_corner_logo.py` / `update_slide2_pipeline.py` | ابزارهای presentation قدیمی |
