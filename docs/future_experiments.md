# Future Experiments — پس از اتمام پروژه اصلی

این فایل شامل آزمایش‌هایی‌ه که عمداً برای بعد از تحویل پروژه‌ی B.Sc. نگه داشته شدن.
هدف: بهبود پروژه برای مقاله یا نسخه‌ی بعدی.

---

## EXP-1: Window Size و Overlap Tuning

**هدف:** پیدا کردن اندازه‌ی بهینه‌ی پنجره و میزان overlap بین پنجره‌ها.

**چرا الان نکردیم:** مدل train شده، golden vectors و HLS code همه بر اساس 1024 هستن.
عوض کردن الان = شروع از صفر.

**configs پیشنهادی:**
| window | stride | overlap | توضیح |
|--------|--------|---------|-------|
| 512 | 512 | 0% | نصف فعلی، نصف compute |
| 512 | 256 | 50% | boundary effects بهتر |
| 1024 | 512 | 50% | فعلی با overlap |
| 256 | 256 | 0% | خیلی کوچیک، سریع |
| 2048 | 2048 | 0% | دو برابر، pattern بلندتر |

**معیار مقایسه:** test_acc، macro_F1، و اندازه‌ی attention matrix برای FPGA.

**نکته FPGA:** هر چی window کوچیک‌تر، attention matrix کوچیک‌تر و tiling ساده‌تر.
با window=512 اگه accuracy زیر 1% افت داشت، ارزش سوئیچ رو داره.

---

## EXP-2: Channel Ablation (کاهش تعداد کانال)

**هدف:** بررسی اینکه با چند کانال کمتر چقدر accuracy افت می‌کنه.
کاربرد عملی: اگه پروژه رو با سنسورهای کمتر بخوایم پیاده‌سازی کنیم چقدر هزینه کم می‌شه.

**configs پیشنهادی:**
| کانال‌ها | تعداد | توضیح |
|----------|-------|-------|
| همه | 8 | فعلی — baseline |
| بدون tachometer | 7 | حذف ch0 (pulse reference) |
| فقط vibration | 6 | ch1-ch6 (بدون tachometer و mic) |
| فقط underhang | 3 | ch1-ch3 (یه bearing) |
| فقط یه محور | 2 | axial و radial از b1 |

**سوال اصلی:** آیا با ۳ accelerometer (یه bearing) می‌شه ۹۵٪+ دقت داشت؟
اگه بله: در پروژه‌ی عملی فقط یه bearing نیاز داری، هزینه نصف می‌شه.

---

## EXP-3: 6-Class Classification (کلاس‌بندی دقیق‌تر)

**هدف:** به‌جای گروه‌بندی، هر نوع خرابی رو جداگانه classify کن.

**کلاس‌های پیشنهادی:**
```
0: normal
1: imbalance
2: horizontal_misalignment
3: vertical_misalignment
4: bearing_underhang_fault
5: bearing_overhang_fault
```

**چرا الان نکردیم:** proposal استاد 4 کلاس گفت.
تغییر = relabeling dataset + retraining + requantization + HLS update.

**نکته:** تعداد کلاس تأثیر تقریباً صفر روی اندازه مدل داره
(فقط آخرین لایه: 16→4 می‌شه 16→6 = 32 param اضافه).

---

## EXP-4: Architecture Mini-Tuning (کوچیک‌تر از F_1layer)

**هدف:** پیدا کردن کوچیک‌ترین مدلی که accuracy قابل قبول داره.

**وضعیت فعلی:**
- F_1layer: d=32, 1layer, ffn=64, **9,492 param**, val 98.28%
- P_prior_like: d=8, 2layer, ffn=8, **1,582 param**, val 88.87% ← خیلی کوچیک

بین این دو یه فضای بزرگ وجود داره. configs پیشنهادی:

| name | d_model | layers | ffn | params تقریبی | توضیح |
|------|---------|--------|-----|---------------|-------|
| G_d16 | 16 | 1 | 32 | ~2,500 | نصف F_1layer |
| H_d24 | 24 | 1 | 48 | ~5,500 | بین G و F |
| I_d32_ffn32 | 32 | 1 | 32 | ~7,500 | FFN کوچیک‌تر |

**سوال:** آیا با ۲,۵۰۰ param هنوز ۹۶٪+ داریم؟
اگه بله: برای FPGA خیلی بهتره.

---

## EXP-5: Normalization Method Comparison

**هدف:** مقایسه روش‌های مختلف normalization.

**چرا z-score per-window انتخاب شد:**
- هر window خودکفاست (برای FPGA ایده‌آل)
- DC offset و amplitude variation رو handle می‌کنه
- در literature برای vibration signals رایجه

**روش‌های دیگه برای مقایسه:**
| روش | مزیت | معایب |
|-----|-------|-------|
| z-score per-window (فعلی) | FPGA-friendly، خودکفا | اگه std≈0 مشکل داره |
| MinMax per-window | [0,1] range مشخص | به outlier حساسه |
| z-score global_train | یکدست‌تر | نیاز به ذخیره آمار در FPGA |
| Robust (median/IQR) | به outlier robust | سخت‌تر برای FPGA |

**نکته:** احتمالاً تفاوت زیادی نداره چون مدل از window-level features یاد می‌گیره.

---

## EXP-6: Train/Val/Test Split Ratio

**هدف:** بررسی تأثیر تقسیم‌بندی متفاوت.

**فعلی:** 70/10/20 با 1951 فایل → train=1365, val=195, test=391

**configs پیشنهادی:**
| train | val | test | توضیح |
|-------|-----|------|-------|
| 70 | 10 | 20 | فعلی |
| 75 | 10 | 15 | test کمتر، train بیشتر |
| 60 | 20 | 20 | val بیشتر برای tuning |
| 80 | 10 | 10 | maximum train |

**نکته:** با accuracy فعلی (98.21%) احتمالاً تفاوت زیادی نمی‌بینیم.
این experiment آخرین اولویته.

---

## اولویت‌بندی

| اولویت | Experiment | دلیل |
|--------|------------|------|
| 1 | EXP-4 (mini architecture) | مستقیم روی FPGA تأثیر داره |
| 2 | EXP-1 (window size) | روی FPGA memory تأثیر داره |
| 3 | EXP-2 (channel ablation) | کاربرد عملی مستقیم |
| 4 | EXP-3 (6 classes) | برای مقاله |
| 5 | EXP-5 (normalization) | احتمالاً تفاوت کمی داره |
| 6 | EXP-6 (split ratio) | کمترین اولویت |

---

_آخرین بروزرسانی: 2026-08-13_
_وضعیت پروژه اصلی در زمان نوشتن این فایل: Stage 7 complete, Stage 8 in progress._
