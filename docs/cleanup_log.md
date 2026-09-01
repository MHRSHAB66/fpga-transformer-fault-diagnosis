# Cleanup Log

---

## پاکسازیِ 2026-09-01 (ادامه) — حذفِ سیستمِ کش

### دلیل
کش disk-based فقط برای اجراهای مکرر local مفید بود.
اجرای نرم‌افزاری روی Kaggle انجام می‌شود (kernel هر بار از صفر)،
و مسیر سخت‌افزاری FPGA جداگانه است.
کش پیچیدگی بی‌فایده اضافه می‌کرد.

### حذف‌شده‌ها

| فایل/مورد | توضیح |
|---|---|
| `src/data/preprocessed_cache.py` | کل ماژول کش — حذف کامل |
| `NORMALIZATION_METHOD` در `config.py` | فقط برای کش key استفاده می‌شد |
| `NORMALIZATION_SCOPE` در `config.py` | فقط برای کش key استفاده می‌شد |
| `preprocessing_cache_enabled/path` در stats | از `build_dataloaders` حذف شد |
| monkey-patch در `evaluate_f1layer.py` | دیگر cache‌ای نیست که disable شود |

### تغییر رفتار
`build_X_y_from_file_index` به جای `load_or_create_preprocessed_array`،
مستقیماً `load_one_csv` + `preprocess_dataframe` را صدا می‌زند.
نتیجه عددی تغییر نمی‌کند — فقط cache hit دیگر نیست.

---

## پاکسازیِ 2026-09-01 — حذفِ normalize_dataframe و متغیرهای اضافه

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

## پاکسازیِ 2026-09-01 (ادامه)

### حذف `global_normalization_stats` و `normalization_fit_seconds` از `make_dataloaders.py`

| چه چیزی | کجا |
|---------|-----|
| `global_normalization_stats = None` (متغیر) | `build_dataloaders` |
| `normalization_fit_seconds = 0.0` (متغیر) | `build_dataloaders` |
| `global_normalization_stats=...` (آرگومان) | سه فراخوانی `build_X_y_from_file_index` |
| `global_normalization_stats: Optional[dict]` (پارامتر) | امضای `build_X_y_from_file_index` |
| `"global_stats"` و `"global_stats_fitted_on"` | `stats` dict |
| `"global_normalization_fit"` | `stats["preprocessing_timing_seconds"]` |

**دلیل:** بازماندهٔ نرمال‌سازیِ `global_train` بودند که در EXP-5 آرشیو شد.
همیشه `None` و `0.0` بودند و هیچ اثری در pipeline نداشتند.

---

## پاکسازیِ 2026-09-01

### حذف `normalize_dataframe` از `src/data/prepare_dataset.py`

| چه چیزی | وضعیت |
|---------|--------|
| `normalize_dataframe` (تابع) | **حذف شد** |
| `preprocess_dataframe(normalize=True)` | حالا `NotImplementedError` می‌دهد |
| `normalize_windows_array` | نگه داشته شد — فعال و در pipeline |

**دلیل:** کدِ یتیم بود. آخرین صدازننده‌اش (`archive/preprocess_data.py`) در کامیتِ
`c1b23b7` آرشیو شده بود. پایپلاینِ فعال همیشه `normalize=False` می‌فرستد.
نرمال‌سازی در `make_dataloaders.py` روی پنجره‌ها انجام می‌شود، نه روی فایل.

**تاریخچه:** `normalize_dataframe` از مسیرِ قدیمیِ استخراجِ ویژگی مانده بود که
در EXP-5 (کامیتِ `b9e00a8`) کنار گذاشته شد.

---

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
