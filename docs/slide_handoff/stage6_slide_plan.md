# Stage 6 Slide Plan — Quantization & Fixed-Point (bilingual content)

> Drop-in content for slides 11–16 of the approved outline. Slides 1–10 are
> covered by `slide_by_slide_plan.md` (software phase). Language: Persian
> narration + English technical terms. Figures are in `stage6_figures/`.
>
> Class map (fixed): 0=normal, 1=imbalance, 2=misalignment, 3=bearing_fault.

---

## Slide 11 — چرا و چگونه کوانتیزیشن؟ / Why & How Quantization

**Title:** آماده‌سازی مدل برای سخت‌افزار — Quantization (Stage 6)

**Bullets:**
- روی FPGA خودِ مدار باید ساخته شود؛ float32 پرهزینه است. مدل به **fixed-point integer** تبدیل می‌شود.
- فرمت انتخابی: **int16 symmetric per-tensor** (`real = int_code × scale`) — ساده‌ترین برای MAC سخت‌افزاری.
- رویکرد **hybrid**: ضرب‌های ماتریسی (matmul) کاملاً int16؛ توابع غیرخطی (softmax, LayerNorm, GELU) با دقت بالا (معادل `hls::exp`/`hls::sqrt`/LUT در Vitis HLS).
- بدون retraining، بدون تغییر معماری، بدون تغییر split.

**Figure:** `stage6_figures/fixed_point_scheme.png`
**Speaker note:** «آبی‌ها int16 هستند، نارنجی‌ها با دقت بالا. این دقیقاً همان چیزی است که HLS می‌دهد.»

---

## Slide 12 — نتایج دقت کوانتیزیشن / Quantization Accuracy Results

**Title:** اثر کوانتیزیشن روی دقت — int16 بدون افت

**Content:** مقایسه روی زیرمجموعه واقعی test (۲۹٬۲۸۰ پنجره، ۱۲۰ فایل):

| Format | Accuracy | Drop |
|---|---|---|
| float32 (baseline) | 98.14% | — |
| **int16 per-tensor** | **98.14%** | **0.00%** |
| int16 per-channel | 98.14% | 0.00% |
| int8 per-tensor | 98.09% | 0.05% |
| int8 per-channel | 98.13% | 0.01% |

- **int16 صفر افت** حتی در ساده‌ترین حالت (per-tensor).
- int8 هم ≤۰.۰۵٪ — گزینه‌ای برای کاهش بیشتر منابع (فاز بعدی/مقاله).

**Figure:** `stage6_figures/quant_accuracy_comparison.png`
**Speaker note:** «int16 را انتخاب کردیم چون حاشیه امن کامل دارد؛ int8 را برای مقایسه منابع نگه می‌داریم.»

---

## Slide 13 — شبیه‌ساز Fixed-Point و Golden Vectors

**Title:** مرجع نرم‌افزاری برای سخت‌افزار — Fixed-Point Simulator & Golden Vectors

**Bullets:**
- کل مسیر inference به‌صورت صریح در Python بازنویسی شد (attention باز شده).
- **اعتبارسنجی:** خروجی شبیه‌ساز با مدل PyTorch اختلاف `2×10⁻⁶` دارد → مرجع دقیق است.
- دقت مسیر int16 کامل: **۹۸.۴۶٪** (صفر افت)، **۹۹.۹۸٪** توافق پیش‌بینی با float.
- **Golden vectors:** ۱۶ پنجره متعادل از ۴ کلاس → **۱۶/۱۶ درست**. این‌ها مرجع تست HLS در Stage 8 هستند (تطابق کلاس دقیق + logit در حد tolerance).

**Figure:** `stage6_figures/golden_validation.png`
**Speaker note:** «این golden vectorها قرارداد صحت سخت‌افزار هستند؛ HLS باید همین‌ها را بازتولید کند.»

---

## Slide 14 — نکته FPGA: تحلیل حافظه / FPGA Insight: Memory

**Title:** یک یافته مهم برای سخت‌افزار — Positional Encoding غالب است

**Bullets:**
- وزن‌های یادگیرنده (int16): فقط **۳۳.۶ KiB** — بسیار کوچک.
- جدول ثابت Positional Encoding (float32): **۱۲۸ KiB** — تقریباً **۴ برابر** کل وزن‌ها!
- نتیجه: بزرگ‌ترین مصرف BRAM، وزن‌ها نیستند بلکه PE است.
- گزینه‌ها برای Stage 8: int16 کردن PE (→۶۴KiB) یا محاسبه on-the-fly با sin/cos.

**Figure:** `stage6_figures/memory_breakdown.png`
**Speaker note:** «این یافته مسیر بهینه‌سازی حافظه سخت‌افزار را مشخص می‌کند و برای مقاله هم جالب است.»

---

## Slide 15 — مسیر پیش‌رو / Roadmap Ahead

**Title:** گام‌های بعدی — Hardware Path

**Bullets:**
- **Stage 7/8:** پیاده‌سازی هسته استنتاج روی FPGA با **Vitis HLS**؛ تطبیق با golden vectors؛ گزارش منابع (LUT/FF/BRAM/DSP) و latency.
- **Stage 10/11:** ارسال نتیجه از FPGA به PC با **UART**؛ رابط کاربری ساده برای نمایش کلاس خرابی.
- **Stage 9 (اختیاری):** ورودی حسگر واقعی (ADXL345) در صورت فراهم‌بودن سخت‌افزار.
- ابتدا HLS برای اتمام سریع‌تر؛ سپس در صورت فرصت RTL برای مقایسه.

**Figure:** `diagrams/stage_status_diagram.png` (به‌روزرسانی: Stage 6 سبز، Stage 7 زرد)
**Speaker note:** «هدف نزدیک: شبیه‌سازی HLS که یک کلاس به‌ازای هر پنجره تولید کند.»

---

## Slide 16 — جمع‌بندی / Summary

**Title:** جمع‌بندی وضعیت پروژه

**What is done:**
- فاز نرم‌افزار کامل: Transformer سبک، **دقت test ۹۸.۲۲٪**، fault recall ۹۹.۷۵٪.
- Stage 6 کامل: **int16 بدون افت دقت**، وزن‌ها export شد، شبیه‌ساز fixed-point ساخته و اعتبارسنجی شد، golden vectors تولید شد.

**Honest limitations (صادقانه):**
- هنوز روی FPGA پیاده نشده؛ منابع و latency واقعی اندازه‌گیری نشده.
- UART و UI هنوز ساخته نشده.
- کوانتیزیشن activation فعلاً dynamic است؛ static scale در Stage 8 نهایی می‌شود.

**Speaker note:** «فاز نرم‌افزار و آماده‌سازی کوانتیزیشن کامل است؛ گام بعدی پیاده‌سازی سخت‌افزار است.»

---

### فایل‌های نمودار Stage 6 (در `docs/slide_handoff/stage6_figures/`)
| فایل | اسلاید |
|---|---|
| `fixed_point_scheme.png` | 11 |
| `quant_accuracy_comparison.png` | 12 |
| `golden_validation.png` | 13 |
| `memory_breakdown.png` | 14 |
