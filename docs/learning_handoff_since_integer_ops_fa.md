# راهنمای کامل ادامه پروژه از `integer_ops.py` تا وضعیت فعلی

**پروژه:** تشخیص عیب ماشین‌آلات با Transformer روی FPGA  
**بازه این یادداشت:** از زمان مرور `src/quantize/integer_ops.py` تا آماده‌شدن بسته پیش از برد  
**آخرین به‌روزرسانی:** ۱۴۰۵/۰۷/۱۰ - 2026-10-02  
**مخاطب:** خودم و هر ChatGPT/مدرسی که قرار است این مسیر را قدم‌به‌قدم برایم توضیح دهد

---

## 0. هدف این فایل چیست؟

من در بخشی از توسعه پروژه حضور ذهن کامل نداشتم و تعداد زیادی آزمایش نرم‌افزاری و سخت‌افزاری در مدت کوتاهی انجام شد. این فایل قرار است:

1. نقطه شروع بحث، یعنی `integer_ops.py`، را توضیح دهد؛
2. تصمیم‌های اصلی را به ترتیب زمانی ثبت کند؛
3. بگوید هر تغییر چرا انجام شد؛
4. نتایج موفق و ناموفق را از هم جدا کند؛
5. فرق «تخمین»، «اندازه‌گیری»، «شبیه‌سازی» و «آزمایش روی برد» را روشن کند؛
6. فایل‌های مهمی را که باید بعداً بخوانم مشخص کند؛
7. به ChatGPT بعدی کمک کند مطالب را کم‌کم و بدون پرش برایم تدریس کند.

این فایل جایگزین گزارش‌های مهندسی دقیق نیست. برای عددهای نهایی باید به artifactها و گزارش‌هایی که در انتهای فایل آمده‌اند مراجعه شود.

---

## 1. خلاصه خیلی کوتاه کل داستان

مسیر پروژه بعد از `integer_ops.py` به‌طور خلاصه این بود:

```text
مدل float یک‌لایه
    -> شبیه‌سازی استنتاج کاملاً صحیح
    -> انتخاب INT10
    -> خروجی گرفتن وزن‌ها، scaleها و بردارهای طلایی
    -> پیاده‌سازی HLS برای SEQ=1024
    -> جا دادن هسته روی 7020 و حتی یک نسخه روی 7010
    -> تلاش‌های زیاد برای کاهش تأخیر SEQ=1024
    -> اثبات اینکه معماری 1024 توکنی با بهینه‌سازی زمان‌بندی به هدف نمی‌رسد
    -> حفظ پنجره 1024 نمونه‌ای ولی تجمیع 4:1 و ساخت مدل SEQ=256
    -> آموزش مجدد مدل 256 توکنی روی Kaggle
    -> کوانتیزه‌سازی و ارزیابی کامل INT10
    -> HLS و تطابق دقیق 16/16
    -> رسیدن به 14.905 ms در برابر بودجه 20.48 ms
    -> اتصال PS + DMA + IP در Vivado
    -> Place & Route موفق و تولید Bitstream/HWH/XSA
    -> ساخت driver، رابط PC و بسته مستقل PYNQ
    -> وضعیت فعلی: از نظر پیش از برد آماده؛ آزمایش فیزیکی هنوز انجام نشده است
```

مهم‌ترین نتیجه فعلی:

- پنجره واقعی ورودی هنوز **1024 نمونه × 8 کانال** و معادل **20.48 ms** است.
- داخل مسیر سخت‌افزار، هر چهار نمونه متوالی به یک توکن تبدیل می‌شوند.
- Transformer جدید **256 توکن** می‌گیرد.
- دقت کامل INT10 روی 95,404 پنجره آزمون: **97.7359%**.
- Macro-F1: **95.6282%**.
- CSim با انواع واقعی `ap_int`: **16/16 دقیق**.
- فاصله خروجی زمان‌بندی‌شده: **1,490,527 cycle = 14.90527 ms در 100 MHz**.
- سیستم کامل PS + DMA + Transformer بعد از مسیریابی timing را پاس کرده است.
- این نتیجه هنوز **اجرای واقعی روی برد نیست**.

---

## 2. نقطه شروع: `src/quantize/integer_ops.py`

### 2.1 چرا این فایل مهم بود؟

مدل PyTorch به‌طور طبیعی با عددهای اعشاری کار می‌کند، ولی پیاده‌سازی مستقیم محاسبات float روی FPGA منابع زیادی مصرف می‌کند و سنتز را کند یا غیرعملی می‌کند. بنابراین به یک مرجع نرم‌افزاری نیاز داشتیم که رفتار مورد انتظار سخت‌افزار صحیح را بازسازی کند.

`integer_ops.py` ابزارهای پایه این مرجع را فراهم می‌کند:

- کوانتیزه‌سازی متقارن بدون zero-point؛
- dequantization برای مقایسه و تحلیل؛
- requantization صحیح بعد از ضرب و جمع؛
- نمایش multiplier حقیقی به‌شکل multiplier صحیح و shift؛
- LUT برای `GELU`؛
- LUT برای `exp` در softmax؛
- تقریب fixed-point برای reciprocal square root در LayerNorm.

### 2.2 معنی «integer-only» در این پروژه

منظور این نیست که هیچ عدد اعشاری در مرحله آماده‌سازی وجود ندارد. منظور این است که:

- tensorهایی که هنگام inference داخل شبکه حرکت می‌کنند، صحیح هستند؛
- scaleها و multiplierها از قبل و به‌صورت offline محاسبه می‌شوند؛
- ضرایب حقیقی به ثابت‌های صحیح و shift تبدیل می‌شوند؛
- توابعی مثل `exp` و `GELU` در زمان اجرا محاسبه float ندارند و از جدول استفاده می‌کنند.

### 2.3 توابع کلیدی که باید بعداً دقیق یاد بگیرم

#### `qint_max(bits)`

بیشترین مقدار مثبت قالب متقارن را می‌دهد:

```text
int8  -> 127
int10 -> 511
int16 -> 32767
```

برای INT10 بازه مورد استفاده پروژه `[-511, +511]` است.

#### `quantize_sym(x, scale, bits)`

فرمول اصلی:

```text
q = clip(round(x / scale), -qmax, +qmax)
```

یعنی float به code صحیح تبدیل می‌شود.

#### `quantize_multiplier(m)` و `apply_multiplier(...)`

برای برگرداندن accumulator بزرگ به محدوده activation از ضرب صحیح و shift استفاده می‌شود. این همان requantization است. هدف این دو تابع این است که ضریب حقیقی به فرم قابل ذخیره در سخت‌افزار تبدیل شود.

#### LUTهای `GELU`، `exp` و `rsqrt`

این توابع غیرخطی برای هر ورودی ممکن یا برای یک بازه کنترل‌شده از قبل محاسبه می‌شوند. FPGA در زمان inference فقط index جدول را می‌سازد و مقدار صحیح را می‌خواند.

### 2.4 تصمیم اصلی بعد از مرور این فایل

فقط وزن‌ها کوانتیزه نشدند. یک **مدل مرجع استنتاج صحیح** ساخته شد تا:

1. تمام tensorهای میانی رفتار integer داشته باشند؛
2. خروجی Python با HLS قابل مقایسه باشد؛
3. golden vector تولید شود؛
4. اثر bit-width بر دقت اندازه‌گیری شود.

---

## 3. انتخاب مدل مرجع برای FPGA

### 3.1 چرا بهترین مدل نرم‌افزاری انتخاب نشد؟

یک Transformer دو‌لایه در آزمایش نرم‌افزاری حدود **98.22%** دقت داشت، اما مدل منتخب FPGA نسخه کوچک‌تر `F_1layer` شد:

- یک encoder layer؛
- `d_model = 32`؛
- `nhead = 4`؛
- `dim_feedforward = 64`؛
- چهار کلاس خروجی؛
- حدود 9,492 پارامتر.

علت انتخاب: هدف پروژه فقط بیشینه‌کردن دقت نبود؛ مدل باید قابل کوانتیزه‌سازی و قابل نگاشت روی FPGA نیز می‌بود.

### 3.2 دقت نسخه یک‌لایه اصلی

برای مسیر قدیمی با طول دنباله 1024:

- float test accuracy: حدود **97.55%**؛
- INT10 test accuracy: حدود **97.11%**؛
- ارزیابی روی کل **95,404** پنجره آزمون انجام شد.

### 3.3 چرا INT10 انتخاب شد؟

bit-widthهای مختلف بررسی شدند. INT8 منابع کمتری می‌خواست، اما کلاس imbalance به‌شکل محسوسی آسیب دید و دقت کلی نیز پایین آمد. INT10 توازن قابل‌قبولی بین دقت و هزینه سخت‌افزاری ایجاد کرد.

تصمیم نهایی آن مرحله:

```text
activation = 10 bit
weight     = 10 bit
attention  = 10 bit
accumulatorهای میانی = پهن‌تر، متناسب با عملیات
```

نکته مهم: ده‌بیتی‌بودن ورودی و وزن به این معنی نیست که تمام accumulatorها نیز 10 بیت باشند. جمع تعداد زیادی حاصل‌ضرب به بیت‌های بیشتری نیاز دارد.

---

## 4. export سخت‌افزاری و golden vectorها

بعد از تثبیت مدل و قالب عددی، موارد زیر تولید شدند:

- وزن‌ها و biasهای صحیح؛
- scaleها و multiplierهای requantization؛
- LUTهای توابع غیرخطی؛
- positional encoding آماده‌شده؛
- manifest شامل شکل tensorها و ثابت‌ها؛
- ورودی‌های طلایی؛
- خروجی صحیح مورد انتظار برای هر ورودی طلایی.

هدف golden vector این است که Python و C++/HLS دقیقاً روی یک ورودی کار کنند. اگر خروجی‌ها برابر نباشند، هنوز حق نداریم درباره صحت سخت‌افزار ادعا کنیم.

---

## 5. اولین مسیر HLS با `SEQ=1024`

### 5.1 مشکل اولیه منابع و زمان سنتز

اولین پیاده‌سازی HLS با کل طول 1024 توکن مشکلات زیادی داشت:

- bufferهای بزرگ `SEQ × D` و `SEQ × DFF`؛
- نگه‌داری `q`، `k`، `v` و context برای کل دنباله؛
- برخی مسیرهای `long` یا `double`؛
- حلقه attention با پیچیدگی تقریبی `O(SEQ²)`؛
- زمان زیاد C-synthesis، مخصوصاً در توابع float و حلقه attention؛
- گزارش‌های loop flatten که warning بودند، نه لزوماً خطای عملکردی.

در بعضی اجرای اولیه، Vitis HLS مدت بسیار زیادی در scheduling باقی می‌ماند. این مسئله باعث شد مسیر float و ساختارهای materialized بازبینی شوند.

### 5.2 اصلاح نوع داده و حذف محاسبات اعشاری زمان اجرا

برای کاهش هزینه:

- مسیرهای float/double از datapath حذف شدند؛
- requantization به ضرب صحیح + shift تبدیل شد؛
- نوع bufferها به `ap_int`های مشخص تغییر کرد؛
- Q40 و accumulatorهای پهن فقط در جاهایی استفاده شدند که برای جلوگیری از overflow لازم بود؛
- خروجی HLS با مرجع صحیح دوباره تطبیق داده شد.

### 5.3 streaming و tile کردن

هدف این بود که تمام tensorهای میانی برای 1024 توکن هم‌زمان نگه‌داری نشوند. بااین‌حال attention را نمی‌توان صرفاً به چهار قطعه مستقل شکست، زیرا هر query باید به همه key/valueها دسترسی داشته باشد.

راه‌حل استفاده‌شده در نسخه‌های بهتر:

- نگه‌داری K و V لازم؛
- پردازش queryها به‌صورت ردیفی/streaming؛
- جلوگیری از materialize کردن score کامل برای همه queryها؛
- reuse یا کوچک‌سازی bufferهای میانی؛
- اشتراک برخی واحدهای محاسباتی.

---

## 6. بررسی 7020 و 7010 برای نسخه 1024 توکنی

این دو قطعه در آن زمان **هدف مقایسه مهندسی** بودند، نه انتخاب نهایی برد.

### 6.1 نسخه `v3` روی 7020

نسخه BRAM-fixed یا `v3` روی `xc7z020clg400-1` در تخمین C-synthesis جا می‌شد:

| مورد | مقدار |
|---|---:|
| latency | 44,930,707 cycle |
| زمان در 100 MHz | حدود 449 ms |
| BRAM_18K | 108 / 280 |
| DSP | 46 / 220 |
| FF | 23,736 / 106,400 |
| LUT | 26,857 / 53,200 |

از نظر منابع جا می‌شد، اما بسیار کندتر از بودجه 20.48 ms بود.

### 6.2 همان نسخه روی 7010

مصرف مطلق تقریباً همان بود، ولی ظرفیت قطعه کمتر بود:

| مورد | مصرف / ظرفیت 7010 |
|---|---:|
| BRAM_18K | 108 / 120 = 90% |
| DSP | 46 / 80 |
| FF | 23,736 / 35,200 |
| LUT | 26,857 / 17,600 = 152.6% |

نتیجه: نسخه `v3` به‌علت LUT روی 7010 جا نمی‌شد.

### 6.3 کمپین جا دادن روی 7010

چند variant ساخته شد:

- **A:** اشتراک requant خروجی context؛
- **B:** اشتراک بیشتر LayerNorm و مسیرهای غیرخطی؛
- **C:** کاهش context lane به 4؛
- **D:** کاهش context lane به 2؛
- **E:** اشتراک requant و residual و arithmeticهای بزرگ.

نتیجه مهم:

- کاهش laneها LUT را آن‌طور که انتظار می‌رفت کم نکرد و فقط latency را زیاد کرد؛
- عامل مؤثر اصلی **operator sharing** بود؛
- Variant E در تخمین C-synthesis روی 7010 جا شد.

نتیجه Variant E:

| مورد | مقدار |
|---|---:|
| latency | 49,129,442 cycle = حدود 491 ms |
| BRAM_18K | 107 / 120 |
| DSP | 36 / 80 |
| FF | 11,905 / 35,200 |
| LUT | 15,138 / 17,600 |

این موفقیت فقط «جا شدن هسته» بود. هنوز real-time نبود و board implementation هم انجام نشده بود.

---

## 7. تلاش برای سریع‌کردن `SEQ=1024`

### 7.1 تحلیل bottleneck

گزارش hierarchy نشان داد بخش query loop بیش از 93% زمان نسخه v3 را مصرف می‌کند. در score calculation، II برابر 4 بود؛ علت اصلی کمبود port خواندن K بود.

### 7.2 T1: بانک‌بندی K

K به هشت بانک cyclic تقسیم شد تا هشت lane در هر cycle قابل خواندن باشند.

نتیجه:

- score loop از `II=4` به `II=1` رسید؛
- interval از حدود 44.93M به حدود 32.84M cycle کاهش یافت؛
- CSim همچنان 16/16 دقیق بود؛
- هنوز حدود 16 برابر کندتر از هدف real-time بود.

### 7.3 T2: اشتراک requant

واحد requant خروجی context مشترک شد:

- latency تقریباً همان T1 باقی ماند؛
- حدود 3,841 LUT و 3,881 FF آزاد شد؛
- برای ادامه parallelization حاشیه بیشتری ایجاد شد.

### 7.4 DATAFLOW و موازی‌کردن headها

نسخه‌های مختلفی آزمایش شدند:

- DATAFLOW اصلاح‌شده؛
- K/V factor 2؛
- K/V factor 2 + DATAFLOW؛
- دو head صریح؛
- چهار head صریح؛
- attention دو مسیره + FFN چهار مسیره.

نتایج منتخب:

| نسخه | interval | زمان در 100 MHz | نتیجه |
|---|---:|---:|---|
| T2 | 32,863,881 | 328.64 ms | شروع این مرحله |
| DATAFLOW اصلاح‌شده | 26,811,336 | 268.11 ms | بهتر، ولی کافی نیست |
| K/V×2 + DATAFLOW | 25,762,760 | 257.63 ms | بهتر، ولی کافی نیست |
| دو head صریح | 9,835,467 | 98.35 ms | بهبود بزرگ |
| چهار head صریح | 9,835,467 | 98.35 ms | بهتر از دو head نشد |
| attention×2 + FFN×4 | 7,212,718 | 72.13 ms | سریع‌تر، نزدیک محدودیت منابع |

چرا چهار head از دو head بهتر نشد؟ چون bottleneck از headها به مراحل دیگری منتقل شده بود. افزایش parallelism وقتی مرحله دیگری محدودکننده است، لزوماً throughput را بهتر نمی‌کند.

### 7.5 بهینه‌سازی K/V prefix

دو نسخه دیگر ساخته شدند:

- `prefix_opt1`: unroll بیشتر؛ نتیجه بدتر شد و رد شد.
- `prefix_opt2`: projectionهای K و V در یک حلقه مشترک ادغام شدند.

`prefix_opt2` بهترین نسخه 1024 توکنی شد:

| مورد | مقدار |
|---|---:|
| interval | 6,229,677 cycle |
| زمان | 62.30 ms |
| BRAM_18K | 216 |
| DSP | 160 |
| FF | 46,839 |
| LUT | 49,889 |
| CSim | 16/16 دقیق |

اما هنوز:

```text
6,229,677 / 2,048,000 = 3.04x
```

یعنی بیش از سه برابر بودجه real-time بود.

### 7.6 آزمایش K/V concurrent DATAFLOW که رد شد

اجرای هم‌زمان projectionهای K و V عملاً latency را بهتر نکرد:

- interval تقریباً همان 6,229,681 cycle؛
- BRAM به 296 رسید، درحالی‌که ظرفیت 7020 برابر 280 بود.

پس هم از نظر زمان فایده نداشت و هم از ظرفیت BRAM عبور کرد.

### 7.7 تحلیل کاهش passهای softmax

بررسی شد که softmax پنج‌مرحله‌ای به سه یا دو pass تبدیل شود.

- نسخه 3-pass می‌توانست bit-exact باقی بماند؛
- نسخه 2-pass و online softmax ترتیب rounding را تغییر می‌دادند و approximate بودند؛
- حتی بهترین تحلیل به حدود 5.895M cycle می‌رسید؛
- bottleneck بعدی `finish_token` می‌شد؛
- این بهبود هنوز 2.88 برابر کندتر از هدف بود.

بنابراین قبل از صرف زمان زیاد برای synthesis، این مسیر با یک gate تحلیلی رد شد.

### 7.8 تصمیم کلیدی این مرحله

نتیجه روشن شد:

> با معماری 1024 توکنی، بهینه‌سازی زمان‌بندی به‌تنهایی برای رسیدن به 20.48 ms کافی نیست.

تغییر اصلی لازم بود، اما نمی‌خواستیم بازه فیزیکی پنجره یا هشت کانال ورودی را حذف کنیم.

---

## 8. تصمیم جدید: حفظ پنجره 1024 نمونه‌ای و ساخت 256 توکن

### 8.1 ایده

به‌جای اینکه پنجره کوتاه شود، چهار نمونه متوالی هر کانال با هم تجمیع می‌شوند:

```text
ورودی acquisition: [1024, 8]
aggregation factor: 4
ورودی Transformer:  [256, 8]
```

در نتیجه:

- بازه زمانی سیگنال همان 20.48 ms باقی می‌ماند؛
- همه هشت کانال حفظ می‌شوند؛
- attention از نظر تعداد توکن بسیار کوچک‌تر می‌شود؛
- مدل نرم‌افزاری باید دوباره آموزش ببیند، چون positional encoding و توزیع tokenها تغییر کرده است.

### 8.2 فایل aggregation

فایل جدید:

```text
src/data/token_aggregation.py
```

دو قرارداد دارد:

1. مسیر float برای آموزش؛
2. مسیر صحیح برای HLS.

قاعده صحیح تقسیم بر چهار:

```text
s = a + b + c + d
q = sign(s) * ((abs(s) + 2) >> 2)
```

این round-half-away-from-zero است و در Python و HLS یکسان پیاده‌سازی شد.

### 8.3 اصلاح dtype

نسخه اولیه aggregation با float64 خروجی می‌داد و ممکن بود DataLoader را با مدل float32 ناسازگار کند. اصلاح شد تا:

- accumulation دقیق در float64 انجام شود؛
- خروجی پیش‌فرض float32 باشد؛
- `factor=1` بدون تغییر مقدار و dtype عمل کند.

### 8.4 flag در DataLoader

در `build_dataloaders` پارامتر زیر اضافه شد:

```python
aggregation_factor=1
```

- مقدار 1 مسیر قدیمی SEQ=1024 را دست‌نخورده نگه می‌دارد؛
- مقدار 4 مسیر SEQ=256 را فعال می‌کند؛
- factor باید توان دو و مقسوم‌علیه window size باشد؛
- `aggregation_factor` و `model_seq_len` در stats ثبت می‌شوند.

---

## 9. آموزش دوباره مدل 256 توکنی روی Kaggle

### 9.1 چرا آموزش مجدد لازم بود؟

اگر فقط HLS را از 1024 به 256 تغییر می‌دادیم، سخت‌افزار دیگر همان مدلی را اجرا نمی‌کرد که آموزش دیده بود. بنابراین مدل جدید باید دقیقاً با preprocessing جدید آموزش می‌دید.

### 9.2 warm-start و scratch

دو روش در طراحی آزمایش در نظر گرفته شد:

- warm-start از checkpoint مدل یک‌لایه 1024 توکنی؛
- آموزش از صفر یا scratch.

پارامترهای learned به طول sequence وابسته نبودند، اما buffer مربوط به positional encoding شکل متفاوت داشت. در warm-start باید کلید `positional_encoding.pe` حذف می‌شد تا buffer جدید 256تایی ساخته شود.

انتخاب نهایی با validation macro-F1 انجام شد، نه test set. نتیجه ثبت‌شده:

| نامزد | بهترین validation macro-F1 | epoch منتخب |
|---|---:|---:|
| warm-start | 0.953825 | 11 |
| scratch | **0.963423** | 22 |

پس نامزد **scratch** انتخاب شد و test set فقط یک بار بعد از انتخاب نهایی ارزیابی شد. checkpoint منتخب در این مسیر ذخیره شده است:

```text
artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt
```

### 9.3 نتیجه float/مرجع

مدل 256 توکنی نشان داد که aggregation چهاربه‌یک اطلاعات طبقه‌بندی را به‌شکل جدی از بین نبرده است. نتیجه دقیق مسیر نهایی صحیح در بخش بعد آمده است.

---

## 10. کوانتیزه‌سازی و ارزیابی کامل مدل SEQ=256

### 10.1 تفاوت دو ترتیب عددی

دو ترتیب ممکن وجود داشت:

```text
A) normalize -> float aggregate -> quantize
B) normalize -> quantize INT10 -> integer aggregate
```

مسیر سخت‌افزار نهایی **B** است، زیرا ورودی transport شده code صحیح است و aggregation نیز داخل IP انجام می‌شود.

### 10.2 نتیجه کامل مسیر واقعی سخت‌افزار

مسیر نهایی:

```text
normalized [1024,8]
    -> quantize INT10 با scale ثابت
    -> aggregate integer 4:1
    -> [256,8]
    -> integer Transformer
    -> 4 logits صحیح
    -> argmax class
```

نتیجه روی کل test split:

| معیار | مقدار |
|---|---:|
| تعداد پنجره | 95,404 |
| پیش‌بینی صحیح | 93,244 |
| accuracy | 0.9773594399 = 97.7359% |
| macro precision | 95.9593% |
| macro recall | 95.3573% |
| macro F1 | 95.6282% |

Recall کلاس‌ها:

| کلاس | Recall |
|---|---:|
| normal | 86.8852% |
| imbalance | 97.8346% |
| misalignment | 99.1513% |
| bearing fault | 97.5581% |

نکته: کلاس normal ضعیف‌ترین recall را دارد، ولی هیچ کلاس collapse نکرده است.

### 10.3 ثابت مهم ورودی

```text
input_scale = 0.028626115354773118
INT10 clamp = [-511, +511]
```

---

## 11. export جدید HLS برای SEQ=256

مسیر جدید به‌صورت مستقل ساخته شد:

```text
hls/f1_int10_seq256/
```

شامل موارد زیر است:

- منبع HLS؛
- weights header؛
- manifest؛
- golden vectorهای مدل 256 توکنی؛
- golden vectorهای ورودی واقعی 1024 نمونه‌ای؛
- Tclهای CSim و synthesis؛
- reportها.

### 11.1 CSim

Vitis HLS 2020.2 با نوع‌های واقعی `ap_int` اجرا شد:

```text
16 / 16 exact
```

این یعنی خروجی HLS برای 16 ورودی طلایی دقیقاً با خروجی مرجع integer Python برابر بود.

این نتیجه چه چیزی را اثبات نمی‌کند؟

- fit نهایی روی board را اثبات نمی‌کند؛
- timing بعد از route را اثبات نمی‌کند؛
- DMA واقعی را اثبات نمی‌کند؛
- سرعت واقعی روی برد را اثبات نمی‌کند.

### 11.2 C-synthesis

برای مسیر نهایی ingress1024 + aggregation + Transformer:

| مورد | تخمین HLS |
|---|---:|
| latency | 1,490,526 cycle |
| interval | 1,490,527 cycle |
| زمان در 100 MHz | 14.90527 ms |
| estimated clock | 7.299 ns |
| BRAM_18K | 145 |
| DSP | 156 |
| FF | 46,948 |
| LUT | 52,864 |

از نظر زمان‌بندی HLS:

```text
14.90527 ms < 20.48 ms
```

پس شرط **real-time محاسباتی** برقرار شد.

### 11.3 وضعیت روی 7010

نسخه SEQ=256 فعلی برای 7010 مناسب نیست و منابع آن از ظرفیت 7010 عبور می‌کنند. هدف عملی پیاده‌سازی کامل به مسیر آزمایشی 7020 منتقل شد.

### 11.4 scoreهای attention در پیاده‌سازی نهایی دقیقاً چگونه نگه‌داری می‌شوند؟

این نکته از روی snapshot دقیق همان sourceای بررسی شد که build نهایی DSP-margin با آن ساخته شده است:

```text
hls/f1_int10_seq256/vitis/transformer_top_seq256_ingress1024_dsp_margin.cpp
```

SHA-256 فایل canonical و snapshot ذخیره‌شده کنار نتایج نهایی یکسان است:

```text
25B91D376E6B752A553062EFCA2B93C85B2C5D8278794AA882E28C1D0BA92B9E
```

پاسخ دقیق:

- در نسخه نهایی آرایه `score_row[1024]` وجود ندارد، چون `SEQ` داخل Transformer برابر **256** است؛
- برای هر query و هر attention head، تابع `df_head` یک آرایه کامل `sc[256]` نگه می‌دارد؛
- علاوه بر آن، `num[256]` برای صورت‌های exp و `qattn[256]` برای وزن‌های attention کوانتیزه‌شده نگه‌داری می‌شوند؛
- پس این نسخه score را به tileهای کوچک یا الگوریتم online تبدیل نکرده است؛
- در عین حال، ماتریس کامل score با شکل `[256,256]` نیز ذخیره نمی‌شود؛ فقط **یک ردیف score برای query جاری در هر head** materialize می‌شود و همین buffer برای query بعدی دوباره استفاده می‌شود؛
- چهار head با `DATAFLOW` به‌صورت فرایندهای مستقل فراخوانی می‌شوند، بنابراین هر head فضای محلی مربوط به ردیف خودش را دارد؛
- K و V نیز برای کل 256 token ذخیره شده و به چهار head جدا شده‌اند: برای هر head آرایه‌هایی با شکل `[256,8]` وجود دارد.

پردازش یک query در هر head پنج گذر اصلی روی طول 256 دارد:

```text
Pass 1: محاسبه scoreها و نوشتن sc[256]
Pass 2: خواندن sc و یافتن rowmax
Pass 3: خواندن sc، lookup کردن exp، نوشتن num[256] و جمع denom
Pass 4: خواندن num، تقسیم بر denom و نوشتن qattn[256]
Pass 5: خواندن qattn و V و جمع‌کردن context
```

محل کد:

- تعریف `sc[SEQ]`: خط 257؛
- محاسبه score: خطوط 258 تا 269؛
- pass بیشینه: خطوط 270 تا 276؛
- تعریف و پرکردن `num[SEQ]`: خطوط 277 تا 292؛
- تعریف و پرکردن `qattn[SEQ]`: خطوط 294 تا 301؛
- محاسبه context از `qattn` و V: خطوط 302 تا 322؛
- اجرای چهار head با DATAFLOW: خطوط 345 تا 358.

این ساختار دلیل مهمی برای تمایز دو جمله زیر است:

```text
درست: کل ماتریس attention ذخیره نمی‌شود؛ فقط row جاری ذخیره می‌شود.
نادرست: attention بدون بافر row و کاملاً online اجرا می‌شود.
```

در آزمایش softmax fusion بررسی شد که passها کمتر شوند، اما مسیر نهایی آن تغییر را نپذیرفت؛ چون سود زمانی محدود بود و نسخه‌های دو-pass/online قرارداد rounding فعلی را تغییر می‌دادند.

---

## 12. فرق real-time محاسباتی با real-time واقعی

این یکی از مهم‌ترین مرزهای پروژه است.

### real-time محاسباتی فعلی

از گزارش schedule می‌دانیم core در 100 MHz تقریباً 14.905 ms interval دارد. چون پنجره هر 20.48 ms تشکیل می‌شود، core از نظر محاسبه فرصت کافی دارد.

### real-time واقعی روی برد

باید این موارد هم اندازه‌گیری شوند:

- انتقال 8192 word ورودی از PS به PL؛
- اجرای IP؛
- بازگشت چهار logit؛
- overhead نرم‌افزار و DMA؛
- تکرار پایدار برای چندین window؛
- عدم drop شدن frameها.

این اندازه‌گیری هنوز انجام نشده است. بنابراین عبارت درست فعلی:

> شرط real-time محاسباتی در فرکانس هدف برقرار شده، اما throughput انتهابه‌انتها روی برد هنوز اندازه‌گیری نشده است.

---

## 13. پیاده‌سازی کامل Vivado و تحلیل پس از مسیریابی

بعد از HLS، تنها IP به‌صورت جداگانه بررسی نشد. یک system کامل آزمایشی ساخته شد:

```text
Zynq PS7
    <-> AXI DMA
    <-> AXI interconnect / HP0
    <-> transformer_top HLS IP
```

رابط‌ها 32 بیتی هستند.

### 13.1 قرارداد انتقال

ورودی:

- 1024 sample؛
- 8 channel؛
- مجموع `1024 × 8 = 8192` کلمه int32؛
- ترتیب row-major: ابتدا تمام کانال‌های sample صفر، سپس sample یک و ...؛
- TLAST روی word آخر.

خروجی:

- چهار logit صحیح int32؛
- TLAST روی logit چهارم؛
- کلاس نهایی با `argmax`.

### 13.2 نتیجه timing پس از route

هدف کلاک: 10 ns یا 100 MHz.

| معیار | مقدار |
|---|---:|
| WNS setup | +0.036 ns |
| TNS setup | 0 |
| WHS hold | +0.020 ns |
| THS hold | 0 |
| fully routed nets | 66,275 / 66,275 |
| routing errors | 0 |

پس timing constraint پاس شده است، اما حاشیه setup کوچک است.

### 13.3 منابع سیستم کامل بعد از route

نسخه نهایی margin build:

| منبع | مصرف | ظرفیت | درصد |
|---|---:|---:|---:|
| Slice LUT | 31,687 | 53,200 | 59.56% |
| Slice Register | 30,889 | 106,400 | 29.03% |
| BRAM Tile | 70.5 | 140 | 50.36% |
| DSP48E1 | 210 | 220 | 95.45% |

### 13.4 چرا نسخه DSP-margin ساخته شد؟

نسخه قبلی هر 220 DSP را مصرف می‌کرد و هیچ حاشیه‌ای باقی نمی‌گذاشت. نسخه جدید با محدودیت `-max_dsp 210` ساخته شد.

مقایسه نسخه 210-DSP با build قبلی:

| تغییر | مقدار |
|---|---:|
| DSP | -10 |
| LUT | +795 |
| FF | +286 |
| WNS | از +0.280 به +0.036 ns |
| latency | +1,522 cycle |
| زمان اضافه | +0.01522 ms |

افزایش زمان حدود 0.1022% است و شرط محاسباتی real-time همچنان برقرار می‌ماند. البته این اختلاف را نباید فقط اثر `-max_dsp 210` دانست، چون نسخه margin هم‌زمان از هسته بازبینی‌شده با arithmetic نوع Q40 و reuse شدن token buffer استفاده می‌کند. بنابراین جدول بالا مقایسه دو build کامل است، نه یک آزمایش تک‌متغیره درباره DSP.

### 13.5 خروجی‌های ساخته‌شده

- BIT؛
- HWH؛
- XSA؛
- گزارش timing؛
- گزارش utilization؛
- گزارش route و DRC؛
- manifest و hashها.

این خروجی‌ها متعلق به پیکربندی آزمایشی `xc7z020clg400-1` هستند. برد فیزیکی نهایی هنوز در این evidence انتخاب/تأیید نشده است.

---

## 14. درایور PYNQ، DMA و برنامه PC

### 14.1 driver سمت PYNQ

فایل اصلی:

```text
pynq/fault_classifier.py
```

دو backend دارد:

- `NumpyBackend`: مرجع نرم‌افزاری برای تست بدون برد؛
- `HardwareBackend`: مسیر Overlay + allocate + DMA برای برد.

HardwareBackend:

- buffer ورودی `(8192,) int32` می‌سازد؛
- buffer خروجی `(4,) int32` می‌سازد؛
- receive channel را قبل از send channel مسلح می‌کند؛
- ورودی را row-major می‌فرستد؛
- چهار logit signed را دریافت می‌کند؛
- class را با argmax می‌سازد.

### 14.2 کنترل IP

IP با `ap_ctrl_hs` ساخته شده است. driver می‌تواند از auto-restart استفاده کند:

```text
control register offset = 0x00
ap_start                = 0x01
auto_restart + start    = 0x81
```

### 14.3 پروتکل PC

در سمت PC:

```text
host/frame_protocol.py
host/fault_monitor.py
```

پیام شامل شناسه frame، class و label است. چون core فعلی probability واقعی تولید نمی‌کند، confidence سخت‌افزاری وجود ندارد و نباید softmax نرم‌افزاری به‌عنوان confidence سخت‌افزار گزارش شود.

---

## 15. ایرادهای اولیه بسته deployment و اصلاح آن‌ها

نسخه اولیه بسته deployment چهار ایراد داشت:

1. بعضی resourceها را از مسیر repository می‌خواند و واقعاً standalone نبود؛
2. نام پوشه پروژه `pynq` می‌توانست کتابخانه vendor PYNQ را shadow کند؛
3. نام BIT و HWH یکسان نبود؛
4. اختلاف latency نسخه 220-DSP و 210-DSP اشتباهاً صفر گزارش شده بود.

اصلاح‌ها:

- تمام configها، goldenها و overlayها داخل bundle قرار گرفتند؛
- اجرای board به‌شکل script انجام می‌شود:

```bash
python driver/run_demo.py --bitstream overlay/design_1_wrapper.bit --port /dev/ttyPS0
```

- BIT و HWH هر دو stem یکسان دارند:

```text
design_1_wrapper.bit
design_1_wrapper.hwh
```

- HWH renameشده از نظر SHA-256 با HWH همان build برابر است؛
- hardware path به torch، `src`، dataset یا checkpoint وابسته نیست؛
- اختلاف latency درست ثبت شد: +1,522 cycle یا +0.01522 ms؛
- packaging script و manifest اضافه شد.

### تست‌های bundle

mockهای vendor PYNQ موارد زیر را کنترل کردند:

- وجود فایل‌ها؛
- تطابق basename BIT/HWH؛
- hash HWH؛
- اعتبار manifest؛
- نبود absolute path؛
- یافتن `axi_dma_0` و `transformer_top_0`؛
- خطای روشن در صورت نبود IP یا HWH؛
- شکل و dtype bufferها؛
- ترتیب recv-before-send؛
- serialization row-major؛
- حفظ logitهای منفی؛
- argmax؛
- import مستقل از repository؛
- عدم import شدن torch در hardware path.

در آخرین بررسی، کل test suite برابر **71 test** و همگی موفق بود.

این تست‌ها منطق و قرارداد را بررسی می‌کنند، نه DMA فیزیکی واقعی.

---

## 16. artifactهای نهایی و نقش هرکدام

### مسیر اصلی نتیجه سخت‌افزاری

```text
artifacts/results/seq256_dma_system_dsp_margin_z7020/
```

این پوشه evidence نهایی سیستم پیاده‌سازی‌شده آزمایشی را دارد:

- HLS source snapshot؛
- CSim log؛
- synthesis report؛
- post-route timing؛
- utilization؛
- route status؛
- BIT/HWH/XSA؛
- manifest و hash؛
- summary JSON.

### بسته آماده انتقال به برد

```text
artifacts/deploy/seq256_z7020_pynq/
```

ساختار:

```text
overlay/   BIT + HWH + XSA
driver/    fault_classifier + run_demo + frame_protocol
config/    model manifest + integer config + HWH contract
golden/    16 ورودی و خروجی مرجع
README.md
MANIFEST.sha256
```

### نتیجه دقت کامل

```text
artifacts/experiments/transformer_f1_seq256_pool4/int10_ingress_local/combined.json
```

### summary قابل خواندن با برنامه

```text
artifacts/results/hls_seq256_int10_summary.json
```

### گزارش مهندسی انسانی

```text
docs/reports/hls_seq256_int10_results.md
```

### checklist قبل از برد

```text
docs/reports/seq256_preboard_readiness.md
```

---

## 17. وضعیت فعلی: چه چیزهایی واقعاً تمام شده‌اند؟

### کامل و اندازه‌گیری‌شده در نرم‌افزار

- file-level split و جلوگیری از leakage؛
- مدل یک‌لایه 256 توکنی؛
- آموزش و checkpoint؛
- ارزیابی کامل 95,404 پنجره؛
- مسیر integer ingress؛
- accuracy و macro-F1؛
- export INT10؛
- golden vector؛
- تست aggregation؛
- driver و PC monitor در محیط نرم‌افزاری.

### کامل در شبیه‌سازی/ابزار سخت‌افزار

- HLS CSim دقیق 16/16؛
- HLS C-synthesis؛
- سیستم PS + DMA + IP؛
- synthesis، place و route؛
- timing closure در 100 MHz؛
- resource report؛
- تولید BIT/HWH/XSA؛
- bundle مستقل؛
- mock validation مسیر DMA.

### هنوز انجام نشده

- انتخاب/تأیید مدل دقیق برد نهایی؛
- اعمال board preset و XDC همان برد؛
- program کردن برد؛
- اجرای 16 golden vector روی FPGA فیزیکی؛
- انتقال DMA واقعی؛
- اندازه‌گیری latency و throughput واقعی؛
- UART واقعی بین برد و PC؛
- demo زنده انتهابه‌انتها؛
- مسیر sensor واقعی که فعلاً اختیاری است.

---

## 18. ادعاهایی که نباید اشتباه گفته شوند

### عبارت درست

- «CSim برای 16 بردار با مرجع integer تطابق دقیق دارد.»
- «C-synthesis فاصله 14.90527 ms را تخمین زده است.»
- «سامانه کامل آزمایشی در Vivado روی part عمومی XC7Z020 route شده و timing 100 MHz را پاس کرده است.»
- «شرط real-time محاسباتی برقرار است.»
- «BIT/HWH/XSA تولید شده‌اند.»
- «driver و قرارداد DMA به‌صورت mock/static تست شده‌اند.»

### عبارت نادرست یا زودهنگام

- «مدل روی برد اجرا شده است.»
- «DMA واقعی تست شده است.»
- «latency واقعی روی برد 14.905 ms است.»
- «برد نهایی پروژه 7020 است.»
- «سیستم sensor-to-PC کامل شده است.»
- «confidence توسط FPGA تولید می‌شود.»

---

## 19. چرا تصمیم‌های اصلی منطقی بودند؟

### چرا INT10؟

INT8 به accuracy و به‌خصوص imbalance آسیب زد؛ بیت‌های بالاتر هزینه بیشتر داشتند. INT10 trade-off مناسب بود.

### چرا مدل یک‌لایه؟

دقت آن کمی کمتر از مدل دو‌لایه بود، ولی نگاشت سخت‌افزاری واقع‌بینانه‌تر داشت.

### چرا ابتدا SEQ=1024 را بهینه کردیم؟

قبل از تغییر مدل لازم بود مطمئن شویم مشکل با اصلاح HLS حل نمی‌شود. bottleneck و lower bound اندازه‌گیری/تحلیل شدند.

### چرا aggregation 4:1؟

هم attention و هم پردازش tokenها با طول sequence رشد می‌کنند. کاهش 1024 به 256 هر دو bottleneck را هدف گرفت، درحالی‌که بازه زمانی خام و هشت کانال حفظ شدند.

### چرا آموزش مجدد؟

ورودی مدل و positional structure تغییر کرده بود؛ استفاده مستقیم از مدل قبلی از نظر علمی و فنی درست نبود.

### چرا DSP-margin؟

مصرف 220/220 هیچ حاشیه‌ای برای implementation و تغییرات جزئی باقی نمی‌گذاشت. آزادکردن 10 DSP با هزینه زمانی ناچیز انتخاب امن‌تری بود.

### چرا هنوز نتیجه را board-ready نمی‌نامیم؟

ابزارها مدل timing و route را اثبات می‌کنند، اما رفتار واقعی Linux/PYNQ، DMA، UART، clock و board فقط روی سخت‌افزار فیزیکی مشخص می‌شود.

---

## 20. ترتیب پیشنهادی مطالعه برای اینکه دوباره به پروژه مسلط شوم

### مرحله 1: مبانی عدد صحیح

1. `src/quantize/integer_ops.py`
2. مفهوم scale، code، accumulator و requantization
3. LUTهای GELU/exp/rsqrt

### مرحله 2: مرجع integer مدل

1. فایل‌های `src/quantize/`
2. manifest و integer config
3. تفاوت float model و integer reference

### مرحله 3: aggregation و مدل 256 توکنی

1. `src/data/token_aggregation.py`
2. بخش aggregation در `src/data/make_dataloaders.py`
3. checkpoint و metrics آزمایش `transformer_f1_seq256_pool4`

### مرحله 4: HLS

1. `hls/f1_int10_seq256/README` یا manifest
2. top-level interface
3. aggregation داخل IP
4. attention، LayerNorm، FFN و classifier
5. golden testbench

### مرحله 5: گزارش‌های performance

1. `docs/reports/hls_throughput_study.md`
2. `docs/reports/hls_realtime_continuation.md`
3. `docs/reports/hls_seq256_int10_results.md`

### مرحله 6: integration

1. `pynq/fault_classifier.py`
2. `pynq/run_demo.py`
3. `host/frame_protocol.py`
4. `host/fault_monitor.py`
5. README بسته deploy

### مرحله 7: وضعیت و محدودیت‌ها

1. `docs/reports/seq256_preboard_readiness.md`
2. `docs/project_status.md`
3. گزارش هفتگی شماره 12

---

## 21. سؤال‌هایی که باید بتوانم بعد از مطالعه جواب بدهم

1. چرا qmax برای INT10 برابر 511 است؟
2. فرق quantization و requantization چیست؟
3. چرا accumulator نمی‌تواند همیشه 10 بیت باشد؟
4. LUT چگونه جای exp یا GELU را در FPGA می‌گیرد؟
5. golden vector دقیقاً چه چیزی را اثبات می‌کند؟
6. چرا 16/16 CSim به معنی اجرای موفق روی برد نیست؟
7. چرا مدل دو‌لایه با accuracy بیشتر انتخاب نشد؟
8. چرا SEQ=1024 با وجود fit شدن روی 7020 real-time نبود؟
9. K banking چگونه II را از 4 به 1 رساند؟
10. چرا چهار head موازی از دو head سریع‌تر نشد؟
11. چرا softmax fusion به‌تنهایی کافی نبود؟
12. چرا کاهش sequence نیازمند retraining بود؟
13. فرق «پنجره 1024 نمونه‌ای» با «Transformer 256 توکنی» چیست؟
14. ترتیب نهایی normalize، quantize و aggregate چیست؟
15. چرا result مسیر integer aggregation کمی با float aggregation فرق دارد؟
16. interval چه فرقی با latency دارد؟
17. چرا 14.905 ms را real-time محاسباتی می‌نامیم؟
18. WNS مثبت چه معنایی دارد؟
19. چرا نتیجه post-route از تخمین HLS معتبرتر است؟
20. چرا با وجود تولید bitstream هنوز board validation باقی مانده است؟
21. DMA چه داده‌ای را با چه ترتیبی منتقل می‌کند؟
22. چرا receive باید پیش از send فعال شود؟
23. چرا confidence فعلاً از FPGA نمی‌آید؟
24. چه چیزهایی بعد از تهیه برد باید تست شوند؟

---

## 22. prompt پیشنهادی برای دادن این فایل به ChatGPT

متن زیر را همراه این فایل به ChatGPT بده:

```text
این فایل تاریخچه و وضعیت پروژه کارشناسی من درباره تشخیص عیب ماشین‌آلات با
Transformer روی FPGA است. من بسیاری از مراحل بعد از integer_ops.py را خودم
عمیق نفهمیده‌ام.

می‌خواهم آن را مثل یک مدرس خصوصی و به زبان فارسی به من آموزش بدهی.

قواعد آموزش:
1. از بخش 2 یعنی integer_ops.py شروع کن و فعلاً جلوتر نرو.
2. هر بار فقط یک مفهوم کوچک را توضیح بده.
3. برای هر مفهوم از اعداد واقعی همین پروژه مثال بزن.
4. تفاوت float، integer code، scale، accumulator و fixed-point را دقیق بگو.
5. بعد از هر بخش 2 یا 3 سؤال کوتاه از من بپرس و تا جواب ندادم جلو نرو.
6. اگر جوابم غلط بود، دقیقاً اشتباه مفهومی را اصلاح کن.
7. ادعاهای measured، estimated، simulated و board-pending را مخلوط نکن.
8. هرجا لازم بود مسیر فایل مربوط در repository را بگو تا هم‌زمان کد را باز کنم.
9. مطالب شکست‌خورده را هم توضیح بده تا بفهمم چرا رد شدند.
10. هیچ نتیجه یا عدد جدیدی اختراع نکن؛ فقط از این فایل و فایل‌های پروژه استفاده کن.

در اولین پاسخ فقط این موارد را توضیح بده:
- نقش integer_ops.py در کل pipeline؛
- quantize_sym؛
- scale؛
- علت بازه [-511,+511] در INT10؛
- یک مثال عددی ساده؛
سپس از من سؤال بپرس و متوقف شو.
```

---

## 23. منابع اصلی این یادداشت

اعداد و نتیجه‌های این فایل از مسیرهای زیر جمع‌بندی شده‌اند:

- `src/quantize/integer_ops.py`
- `docs/project_status.md`
- `docs/reports/hls_z7010_vs_z7020_v3_comparison.md`
- `docs/reports/hls_z7010_fitting_campaign.md`
- `docs/reports/hls_throughput_study.md`
- `docs/reports/hls_realtime_continuation.md`
- `docs/reports/hls_seq256_int10_results.md`
- `docs/reports/seq256_preboard_readiness.md`
- `artifacts/results/hls_seq256_int10_summary.json`
- `artifacts/results/seq256_dma_system_dsp_margin_z7020/summary.json`
- `artifacts/experiments/transformer_f1_seq256_pool4/int10_ingress_local/combined.json`
- `artifacts/deploy/seq256_z7020_pynq/README.md`
- `tests/test_hardware_backend_mock.py`

---

## 24. جمع‌بندی نهایی برای خودم

پروژه از یک مدل float صرف عبور کرده و اکنون این زنجیره را دارد:

```text
MAFAULDA window [1024,8]
    -> normalization
    -> INT10 quantization
    -> exact integer aggregation 4:1
    -> Transformer [256,8]
    -> four integer logits
    -> class argmax
    -> DMA/UART protocol
    -> PC monitor
```

از نظر نرم‌افزار، شبیه‌سازی صحیح، synthesis، place & route و بسته deployment کار زیادی انجام شده و evidence ذخیره شده است. نقطه ناتمام اصلی دیگر طراحی الگوریتم نیست؛ **انتخاب و آزمایش روی برد فیزیکی و اندازه‌گیری end-to-end** است.

برای یادگیری دوباره پروژه نباید مستقیم از Vivado یا DMA شروع کنم. باید از `integer_ops.py`، قرارداد عددی و مدل مرجع صحیح شروع کنم و سپس به HLS و integration برسم.
