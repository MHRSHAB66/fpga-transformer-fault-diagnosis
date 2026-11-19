# ادبیات نزدیک و ظرفیت مقاله: بررسی ۲۰۲۶-۱۱-۱۱

## ۱. نزدیک‌ترین کارهای قبلی چه هستند؟

### Takeaway
«ترنسفورمر integer-only روی FPGA» و حتی «ترنسفورمر تشخیص خرابی یاتاقان روی FPGA» قبلاً منتشر شده‌اند. تمایز احتمالی این پروژه باید در مرز کامل محاسبات، هزینه واقعی و صحت قابل بازتولید آن جست‌وجو شود؛ صرف کنار هم گذاشتن این کلیدواژه‌ها نوآوری کافی را ثابت نمی‌کند.

### Cited Findings
- **Zhu و همکاران، Advanced Science، اولین انتشار ۲۲ آوریل ۲۰۲۶، DOI 10.1002/advs.75373**: عنوان کامل Self-Powered Bearing Sensing and Real-Time Fault Diagnosis Enabled by Non-Invasive Triboelectric Sensors and Edge AI Acceleration. منبع اولیه واقعاً 0.108 ms و 4.59 W و حدود 98% را گزارش می‌کند. دستگاه Zynq-7045 است؛ PL ضرب‌های ماتریسی را انجام می‌دهد و اجرای نرم‌افزاری PS در استنتاج مشارکت دارد. وزن و بایاس INT8 هستند؛ متن، تمام مسیر را integer-only اعلام نمی‌کند. توان FPGA **برآورد Vivado** است. داده از حسگر triboelectric اختصاصی، پنجره 2048 و چهار وضعیت NC/IF/OF/IOF می‌آید؛ MAFAULDA نیست. بخش ۵ تقسیم زمانی 2:1:1 را شرح می‌دهد. نقل پشتیبان: “the FPGA power consumption was estimated using Vivado 2023.2”. بنابراین اعداد قبلی شایعه یا صرفاً خروجی aggregator نیستند، اما مقایسه مستقیم سرعت/دقت با پروژه حاضر درست نیست. [متن اصلی، بخش‌های 2.2.1، 5 و 6](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.75373)
- **Ling، Qian و Schiele، IEEE AIoT 2024**: Integer-only Quantized Transformers for Embedded FPGA-based Time-series Forecasting in AIoT، arXiv:2407.11041؛ نسخه جاری v6 مورخ ۲۱ آوریل ۲۰۲۶. Spartan-7 XC7S15، QAT با 4/6/8 بیت و softmax مبتنی بر LUT و تقسیم integer دارند. برای کاهش هزینه، LayerNorm را با BatchNorm و GELU را در معماری با ReLU جایگزین می‌کنند. کاربرد forecasting و داده‌های ترافیک/کیفیت هوا است. بنابراین کامل‌بودن integer-only به‌تنهایی ادعای تازه‌ای نیست؛ حفظ LN پویا در پروژه حاضر می‌تواند محور مقایسه هزینه باشد. [متن اصلی، بخش‌های II-B، III و IV-C](https://arxiv.org/html/2407.11041v6)
- **Ling، Qian، Haßler و Schiele، ISVLSI 2025**: Automating Versatile Time-Series Analysis with Tiny Transformers on Embedded FPGAs، arXiv:2505.17662v5، ۱۹ سپتامبر ۲۰۲۵. چارچوب خودکار QAT، Optuna و تولید VHDL برای شش دیتاست دارد: PeMS، AirU، UCIHAR، WISDM، ALFA و SKAB؛ MAFAULDA ندارد. روی XC7S15 استقرار موفق و روی iCE40 محدودیت منابع گزارش می‌شود. معماری BatchNorm دارد. جدول I برای UCIHAR مقدار 1.034 ms و 0.067 mJ و برای WISDM مقدار 12.04 ms و 0.855 mJ می‌دهد؛ طول ورودی‌ها متفاوت است. **توان جدول برآورد است**، با اختلاف 5.8% نسبت به اندازه‌گیری؛ latency از GHDL است با اختلاف حدود 2% نسبت به سخت‌افزار. نقل پشتیبان: “Power was estimated at a temperature of 28.0”. [متن و پاورقی جدول I](https://arxiv.org/html/2505.17662v5)
- **Liao و همکاران، IEEE TIM جلد 73، سال ۲۰۲۴**: BearingPGA-Net، DOI 10.1109/TIM.2023.3346517، پیش‌چاپ ۳۱ ژوئیه ۲۰۲۳. CNN تک‌لایه با distillation، 2830 پارامتر، ورودی FFT، کمیت‌سازی fixed-point و Kintex-7 XC7K325T است؛ ترنسفورمر نیست. مقاله شتاب بیش از 200 برابر CPU و افت کمتر از 0.4% در F1/recall/precision روی داده اختصاصی را ادعا می‌کند؛ مخزن رسمی جزئیات 16-bit را دارد. [مقاله نویسندگان](https://arxiv.org/abs/2307.16363)، [مخزن رسمی و citation](https://github.com/asdvfghg/BearingPGA-Net)
- **Vieira، Bauler، Rosa و Silva، Mechanical Systems and Signal Processing 258 (2026)، 114640**: Towards a more realistic evaluation of machine learning models for bearing fault diagnosis، DOI 10.1016/j.ymssp.2026.114640؛ arXiv v5 مورخ ۶ ژوئیه ۲۰۲۶. با CWRU، PU، UORED-VAFCLS و HUST نشان می‌دهد حتی condition-wise split می‌تواند ارزیابی خوش‌بینانه ایجاد کند و تفکیک بر اساس یاتاقان فیزیکی را پیشنهاد می‌کند. این نتیجه مستقیماً اثبات نشت پروژه MAFAULDA نیست؛ محدودیت اعتبار file-level split برای تعمیم به ماشین/یاتاقان جدید را روشن می‌کند. [مقاله اصلی](https://arxiv.org/abs/2509.22267)
- **Zhang، Zhou، Ma، Pirttikangas و Yang، TSViT، arXiv:2311.06916v2، اکتبر ۲۰۲۴**: معماری ترکیب convolution و Transformer برای تشخیص خرابی است؛ چکیده دقت نزدیک 100% روی سه دیتاست را گزارش می‌کند. چکیده شواهد استقرار FPGA ارائه نمی‌دهد. PDF جاری حدود 36 MB بود و ابزار نتوانست متن آن را بازیابی کند؛ جزئیات Gaussian noise و تعریف SNR در این بررسی مستقلاً تأیید نشد. [صفحه اولیه نویسندگان](https://arxiv.org/abs/2311.06916)
- **Sobakinskikh و Bilokon، ۲۴ ژوئیه ۲۰۲۶**: Optimizing Transformer Neural Network for Real-Time Outlier Detection on FPGAs، arXiv:2607.22786. کاربرد داده مالی، PYNQ-Z2، محاسبات float و latency بهینه 37.14 µs از گزارش csynth است؛ با classifier چندکاناله INT10 و latency فیزیکی انتهابه‌انتها هم‌تعریف نیست. [متن اصلی، بخش 4.2](https://arxiv.org/html/2607.22786v1)

### Inferences
- مقایسه قابل دفاع باید ستون‌های دیتاست، split، تعداد کانال، طول پنجره، تعداد token، پارامتر، precision، دستگاه، فرکانس، سهم PS/PL، نوع توان و مرز latency را کنار هم بگذارد. از رتبه‌بندی دقت یا latency بین مسئله‌های متفاوت پرهیز شود.
- تمایز مشخص نسبت به نزدیک‌ترین 1D-ViT احتمالاً «کل پیش‌پردازش دیجیتال تا کلاس در PL کوچک‌تر» است، ولی این موضوع باید با جزئیات implementation خودمان اثبات شود. متن Zhu محل تمام nonlinearها را صریح تعیین نمی‌کند؛ نمی‌توان نتیجه گرفت هیچ‌کدام در PL نیستند.

### Gaps
- این جست‌وجوی متمرکز، مرور نظام‌مند همه کارها نیست؛ عبارت‌هایی مانند «اولین در جهان» یا «کم‌مصرف‌ترین» قابل اثبات نیستند.
- برای جدول نهایی، supplementalهای Zhu و جزئیات دقیق ساعت/منابع/تعریف latency آن و پروتکل مستقل داده BearingPGA-Net باید استخراج شوند.

## ۲. با شواهد موجود، چه روایت مقاله‌ای قابل دفاع است؟

### Takeaway
قضاوت پژوهشی: ظرفیت یک مقاله مهندسی سیستم وجود دارد، اما نوآوری و پذیرش هنوز قطعی نیست. روایت قوی‌تر، «پیاده‌سازی کامل و قابل بازتولید و محدودیت‌های واقعی آن» است؛ افزایش accuracy با رأی‌گیری یا AWGN به‌تنهایی جهش پژوهشی محسوب نمی‌شود.

### Cited Findings
- کار Ling از قبل مسیر quantization تا RTL و tradeoff دقت/منابع/انرژی را ارائه می‌دهد؛ بنابراین مقاله حاضر باید چیزی بیش از استقرار یک مدل روی FPGA نشان دهد. [Ling 2025](https://arxiv.org/abs/2505.17662)
- کار Zhu حسگر فیزیکی و آنلاین واقعی دارد؛ پروژه prepared-window نباید ادعای آزمایش حسگر یا اثبات شرایط صنعتی مشابه آن داشته باشد. [Zhu 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.75373)
- تعمیم به داده جدید از همان یاتاقان با تعمیم به یاتاقان/ماشین جدید متفاوت است؛ ادعای reliability باید با مرز واقعی split سازگار بماند. [Vieira 2026](https://arxiv.org/abs/2509.22267)

### Inferences
- **فرضیه مقاله پیشنهادی**: می‌توان طبقه‌بندی ترنسفورمری چندکاناله را با پیش‌پردازش و nonlinearهای integer در PL برد Zynq کوچک اجرا کرد، تطابق عددی و حاشیه زمانی آن را روی سخت‌افزار نشان داد، و هزینه robustness را با معیارهای دقت–زمان تشخیص–انرژی سنجید. این عبارت پیشنهادی است و تأیید مستقل همه شواهد مخزن در حیطه این یادداشت نبوده است.
- **قوی‌ترین افزودنی قابل‌بررسی**: یک نتیجه هم‌طراحی که فرضیه مشخص را جواب دهد: «حفظ/افزودن اطلاعات دامنه یا انتخاب بودجه دقت عددی چگونه normal recall، robustness و منابع/زمان را تغییر می‌دهد؟» این نیازمند baseline یکسان و ablation است؛ اضافه‌کردن feature یا precision جدید مجوز خودکار تغییر مدل اصلی نیست.
- اگر انرژی فیزیکی، مقایسه منصفانه روی CPU همان برد، صحت زنجیره raw-to-class و stress/recovery جمع شود، وزن سیستم‌سازی بیشتر می‌شود. اگر فقط accuracyهای بیشتر از همان test ثابت اضافه شوند، احتمالاً قوت مقاله به همان نسبت زیاد نمی‌شود.

### Gaps
- نمی‌توان بدون مشاهده روش و ارزیابی نهایی، سطح مجله یا احتمال پذیرش تعیین کرد. تمایز نسبت به پیشینه، هنوز «نامزد تمایز» است.
- نتیجه آزمون‌های severity/speed فعلی و اجرای 300k RT بنا بر ورودی هماهنگ‌کننده موجودند؛ در این وظیفه دوباره اجرا یا ممیزی نشده‌اند.

## ۳. چه آزمایش‌های تازه‌ای تا زمان دسترسی به برد ارزش بیشتری دارند؟

### Takeaway
اولویت با آزمایش‌هایی است که بعداً بدون برد قابل بازیابی نیستند و ادعای مشخص مقاله را محکم می‌کنند: انرژی، تنش هم‌زمان، بازیابی، مرز عددی و تأخیر تشخیص. تکرار clean/AWGN یا طولانی‌ترکردن یک RT run با شرایط قبلی به‌تنهایی اولویت بالا ندارد.

### Cited Findings
- ضرورت تفکیک «توان برآوردی» و «اندازه‌گیری فیزیکی» از پاورقی‌های Ling و روش Zhu مشخص است. [Ling، جدول I](https://arxiv.org/html/2505.17662v5)، [Zhu، بخش ۵](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.75373)
- کوچک‌کردن bitwidth لزوماً latency/energy را کم نمی‌کند؛ tradeoff باید اندازه‌گیری شود. [Ling 2024](https://arxiv.org/abs/2407.11041)

### Inferences
1. **انرژی فیزیکی paired**: idle، raw-preproc-only در صورت مسیر آماده، full inference و ARM همان مدل؛ ولتاژ×جریان در زمان یکسان، idle subtraction جدا و انرژی کل جدا، warm-up، چند تکرار و عدم‌قطعیت ابزار. اگر متر موجود نیست، اکنون نبودش ثبت شود و تخمین Vivado measured نام نگیرد.
2. **deadline زیر تنش و حاشیه خدمت**: همان deadline و مرز قبلی را ثابت نگه دارید؛ ترافیک UART/Ethernet، CPU/DDR load، burst ورودی و throttling/دما را هم‌زمان اعمال کنید؛ شمارش frame، high-water backlog، بدترین latency، drop و recovery ذخیره شود. صرف میانگین یا 0 miss محدود، تضمین hard-real-time برای همه شرایط نیست.
3. **مرز عددی پیش‌پردازش**: ورودی صفر/ثابت/نزدیک‌ثابت، یک کانال stuck، DC-offset، gain، sign inversion، rail و clipping و impulsive noise؛ raw-to-class فیزیکی با مرجع دقیق، رفتار σ≈0 و overflow کنترل شود. این آزمون functional است، مدرک accuracy روی ماشین واقعی نیست.
4. **خرابی ورودی و بازیابی سیستم**: DMA timeout قابل کنترل، توقف stream، reset بین frameها، boot تکراری، قطع‌و‌وصل receiver؛ خروجی‌ها نباید stale، تکراری یا با ID اشتباه شوند. پیش از fault injection، طرح امن و رفتار مورد انتظار از interface فعلی تعیین شود.
5. **مرز real-time رأی‌گیری**: دقت پایدار با زمان تشخیص رخداد تفاوت دارد. اگر recording با transition واقعی نداریم، الحاق دو recording فقط آزمون مصنوعی مشخص‌شده است؛ reset رأی‌گیری، delay تا اولین تصمیم پایدار، false alarm بر واحد زمان و افت ناشی از stale vote اندازه‌گیری شود. افزایش k بدون این هزینه می‌تواند عدد accuracy را گمراه‌کننده کند.
6. **Pareto حداقلی روی برد**: اگر configهای آماده و مجاز وجود دارد، ۲–۳ precision/clock معتبر با مدل و split ثابت، post-route منابع/زمان و energy یکسان ثبت شوند؛ config جدید نیازی نیست فقط برای پرکردن جدول ساخته شود. مقایسه normalization در PS با PL باید دقیقاً همان ریاضی و ورودی را حفظ کند.

### Gaps
- هیچ سنسور یا دیتاست دومِ تأییدشده در این مأموریت در دسترس نبود؛ آزمون fabricated input جای اعتبارسنجی دامنه جدید را نمی‌گیرد.
- هر آزمایش تازه‌ای که انتخاب مدل را از test موجود هدایت کند، داده test را به development evidence تبدیل می‌کند؛ انتخاب checkpoint/threshold روی validation و گزارش صریح تعداد آزمایش‌ها ضروری است. اصل منع نشت و ادعاهای دامنه از [Vieira 2026](https://arxiv.org/abs/2509.22267) پشتیبانی می‌شود؛ این پیشنهاد اجرایی از آن استنتاج شده است.
