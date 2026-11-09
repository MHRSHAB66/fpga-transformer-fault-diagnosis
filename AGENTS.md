# AGENTS.md

## Project Definition

**Project title:** Design and Implementation of a Real-Time Machinery Condition Monitoring System Using a Transformer Neural Network on FPGA

**Student/project identity:** Mehrdad Sheikh Abbasi, B.Sc. project, Amirkabir University of Technology.

This project is a practical engineering pipeline from **data to result**:

1. Use MAFAULDA machinery-condition data as the main dataset.
2. Prepare multichannel vibration/time-series windows.
3. Train and evaluate a lightweight Transformer classifier in Python.
4. Simplify, quantize, and convert the selected model for hardware.
5. Implement FPGA inference logic that produces a classification result per input window.
6. Send the FPGA result to a computer.
7. Display the classification result in a simple PC-side interface.

The project is **not only a model-training project**. The final goal is a complete demonstrable system where a Transformer-based fault classifier is prepared for FPGA execution and its output is shown on a computer.

---

## Source-of-Truth Priority

When future code updates conflict, follow this priority order:

1. **MehrdadSheikhAbbasi_Proposal_v5_Final_Signed.pdf** — primary source of truth (signed 1405/05/04; made the reference by the user on 2026-10-08, replacing MehrdadSheikhAbbasi_Proposal_v3_final.pdf).
2. **Project Definition.pdf** — useful supporting definition, but the final proposal overrides it if there is a conflict.
3. Existing working code and generated artifacts — preserve behavior unless an explicit task says otherwise.
4. **Fault Detection in Industrial Machinery.pdf** — background only; do not implement its full thesis pipeline.
5. **INTELLIGENT FAULT DIAGNOSIS OF TYPE AND SEVERITY IN.pdf** — background only; do not implement its sound-only/XGBoost/MFCC pipeline as the main project.

The two extra papers are for context and justification only. They may inspire evaluation notes or literature-review text, but they must not redefine the implementation target.

---

## Core Scope

The required system is a **multiclass fault classifier** for machinery-condition windows.

### Main input

- **All 8 MAFAULDA channels** as a multichannel time-series window. The verified column order is:
  - col 0 — tachometer (rotation-speed pulse reference)
  - col 1 — acc_axial_b1 (underhang bearing, axial)
  - col 2 — acc_radial_b1 (underhang bearing, radial)
  - col 3 — acc_tangential_b1 (underhang bearing, tangential)
  - col 4 — acc_axial_b2 (overhang bearing, axial)
  - col 5 — acc_radial_b2 (overhang bearing, radial)
  - col 6 — acc_tangential_b2 (overhang bearing, tangential)
  - col 7 — microphone (acoustic emission)
- The primary implemented path uses all 8 channels (CHANNEL_SELECTION = "all" in config.py). A 6-channel vibration-only variant (cols 1–6) is configurable via CHANNEL_SELECTION = "vibration_only" but requires retraining.
- The primary implemented path must remain based on the full MAFAULDA multichannel time-series input, not a sound-only or vibration-only feature-engineering pipeline.
- Window length is expected to be configurable, typically **1024** or **2048** samples.
- Data may come from prepared MAFAULDA windows. A real sensor/laboratory input path can be added when hardware is available, but the minimum project target is a working and evaluable pipeline for prepared windows.

### Main output

For each input window, the system should output:

- predicted fault class;
- confidence/probability score when available;
- frame/window identifier when communicating with the PC.

Expected class groups:

- normal / healthy;
- imbalance;
- misalignment;
- bearing fault.

Do not silently change the class taxonomy. Any class-map change must be explicit, documented, and applied consistently to dataset preparation, model output dimensions, reports, saved metrics, and hardware constants.

---

## Non-Negotiable Restrictions

These rules are strict. Future code updates must respect them.

### FPGA target status

- The board is the **MYIR Z-Turn (XC7Z020-1CLG400C, PYNQ 2.4 / Python 3.6)**,
  confirmed by the user and used for every physical validation since 2026-10-08.
- Report resource/timing numbers from the post-route reports of the deployed
  build; keep synthesis parameters configurable.

### Preserve defined logic

- Do not change the current data-splitting logic unless explicitly requested.
- Do not change the class mapping unless explicitly requested.
- Do not change how windows are labeled unless explicitly requested.
- Do not change input tensor shape conventions without updating every dependent module and documenting the migration.
- Do not change the meaning of metrics or evaluation scripts.
- Do not change trained-artifact formats unless backward compatibility or a migration script is provided.
- Do not bypass the model with hard-coded decisions.
- Do not make the PC perform the final classification in the final hardware stage; the FPGA path must produce the classification result.

### Preserve simulator/testbench behavior

If a simulator, RTL testbench, mock FPGA path, or hardware emulator exists:

- do not change its external behavior to make new code easier;
- do not change clocking, reset, valid/ready, FIFO, framing, or output protocol semantics unless explicitly requested;
- do not change golden-vector expectations without regenerating them from the approved Python reference and documenting why;
- do not replace exact fixed-point behavior with approximate floating-point behavior inside hardware-facing tests.

### Do not expand the project into a different research project

- Do not implement the nine-model comparison from the Koutoulas thesis as the project goal.
- Do not switch the main task to CSU gearbox data.
- Do not switch the main pipeline to sound-only diagnosis, MFCCs, XGBoost, ADASYN/SMOTE, or acoustic low-bit experiments from the Spadini paper.
- Do not add a large Transformer that cannot reasonably be mapped to FPGA.
- Do not add complex preprocessing that makes hardware deployment unrealistic.
- Do not optimize only software accuracy while ignoring quantization, latency, memory, and FPGA resource constraints.

### Git and authorship

- Never add Codex as a co-author or author in git commits or GitHub pull requests. Do not include `Co-Authored-By: Codex` or any Anthropic-attributed trailer lines in commit messages or PR descriptions.

### Engineering discipline

- Do not perform broad refactors unless the user asks for them.
- Prefer small, staged changes.
- Keep config-driven constants in `src/config.py` or the existing config system.
- Maintain runnable scripts and clear command lines.
- Add comments only where they clarify non-obvious project logic.
- Keep dependencies minimal and compatible with `requirements.txt`.
- Never delete existing artifacts or code paths without user approval.

---

## Known Current Codebase Status

This status is based on the last known project state. Before editing, inspect the repository and verify exact filenames.

### Completed work

- Repository/environment scaffolding exists.
- Core Python source structure exists under `src/`.
- MAFAULDA raw data path/configuration has been handled.
- File-level dataset handling has been implemented.
- Basic preprocessing and windowing logic exists, but Stage 2 is not complete because additional research-informed preprocessing experiments remain.
- DataLoader construction exists.
- Data quality checks exist.
- At least one baseline model path exists.
- Transformer baseline implementation exists.
- Training/evaluation artifacts have been generated in `artifacts/` in prior runs.

### Known or expected files/modules

Verify before editing, but previous codebase context mentioned files such as:

- `src/config.py`
- `src/data/load_data.py`
- `src/data/inspect_data.py`
- `src/data/prepare_dataset.py`
- `src/data/check_data_quality.py`
- `src/data/make_dataloaders.py`
- `src/models/mlp_baseline.py`
- `src/models/transformer_baseline.py`
- `src/train_baseline.py`
- `src/train_transformer_baseline.py`
- `src/evaluate/evaluate_model.py`
- `artifacts/checkpoints/`
- `artifacts/results/`

There may be older duplicate or experimental paths. Treat the main project path as:

```text
src/config.py
  -> src/data/load_data.py
  -> src/data/make_dataloaders.py
  -> src/models/transformer_baseline.py
  -> training script
  -> evaluation script
  -> artifacts/checkpoints and artifacts/results
```

Feature-extraction-only or alternative preprocessing paths are experimental unless the user explicitly promotes them.

---

## Stage Plan

### Stage 0 — Project setup and repository organization

**Status:** COMPLETE

**Goal:** Create a runnable Python project structure with dependencies, config, source directories, and artifact directories.

**Expected code scope:**

- `README.md`
- `requirements.txt`
- `.env` / environment path handling
- `src/config.py`
- `data/`, `artifacts/`, and source subdirectories

**Definition of done:**

- Project imports work.
- Configuration can locate the project root and dataset root.
- Runtime artifacts have a defined place.
- No hard-coded user-specific paths are required for normal use.

**Restrictions:**

- Do not scatter constants across scripts.
- Do not introduce new environment requirements without updating documentation.

---

### Stage 1 — MAFAULDA data discovery and class definition

**Status:** COMPLETE

**Goal:** Understand the MAFAULDA folder/file structure, identify usable CSV files, and map raw classes to the project-level classes.

**Expected code scope:**

- data inspection scripts;
- dataset indexing;
- class mapping;
- basic quality checks.

**Definition of done:**

- Raw files are discoverable.
- Class labels are assigned at file level.
- The intended project classes are clear.
- Invalid/malformed files are handled or reported.

**Restrictions:**

- Do not mix MAFAULDA subclasses into new labels without explicit approval.
- Do not change the project from four-class grouped classification to detailed severity classification unless explicitly requested.
- Do not use CSU data in this project path.

---

### Stage 2 — Preprocessing, windowing, and DataLoaders

**Status:** COMPLETE (preprocessing decisions D004–D006 and D017 in `docs/DECISION_LOG.md`)

**Goal:** Convert raw MAFAULDA CSV sequences into model-ready windows while preventing data leakage, and finish the remaining preprocessing experiments that can improve signal quality and FPGA feasibility. Basic loading/windowing/DataLoader logic may already exist, but this stage is not complete until the final preprocessing configuration is selected, documented, and compared against the current basic pipeline.

**Expected code scope already present or expected:**

- CSV loading;
- signal/channel selection;
- normalization or standardization;
- fixed-length windowing;
- file-level train/validation/test split;
- PyTorch Dataset/DataLoader construction.

**Remaining preprocessing work to consider and implement carefully:**

- **Frequency-spectrum analysis before resampling:** inspect FFT/frequency content of MAFAULDA channels and choose a justified resampling/downsampling rate. Do not blindly downsample; preserve fault-relevant frequency bands and document the chosen rate.
- **High-frequency/noise handling:** evaluate whether high-frequency bands are useful signal or mostly noise. If filtering is added, make it configurable and compare metrics with and without filtering.
- **Denoising / white-noise reduction:** add optional denoising such as Wiener filtering or an equivalent simple filter only as a configurable experiment. Do not permanently delete signal content unless evaluation shows it helps.
- **Normalization / scaling variants:** compare the current normalization with alternatives such as per-channel standardization, MinMax scaling, or robust scaling. Fit scaling statistics only on the training split and reuse them for validation/test.
- **Low-energy or silence-like segment handling:** if a sound/microphone channel or real sensor stream is used, optionally test silence/low-energy interval removal. For vibration channels, treat this as low-information segment handling, not automatic audio-style silence deletion.
- **Window length and stride/overlap study:** compare 1024 and 2048 samples, and optionally other lengths/strides that remain compatible with the Transformer and later FPGA buffer design.
- **Class imbalance handling:** keep class distribution reporting. If oversampling or augmentation is used, apply it only to the training split and never to validation/test. Do not make synthetic windows the main evaluation source.
- **Preprocessing configuration export:** save the final preprocessing parameters in a JSON/YAML/config artifact so the same logic can be used by training, evaluation, quantization, golden-vector generation, and hardware input formatting.

**Definition of done:**

- Windows are generated consistently from raw files.
- Train/validation/test separation happens at file level, not merely at window level.
- Tensor shapes are stable and documented.
- Class distribution can be inspected.
- DataLoader timing/sample-count logs can be produced.
- At least one frequency-guided resampling decision is tested and documented.
- Noise/denoising or filtering experiments are either implemented and evaluated, or explicitly rejected with a reason.
- The final normalization/scaling method is selected and uses training-only fitted statistics.
- The final window length/stride is selected and documented for both software and later FPGA buffering.
- The selected preprocessing pipeline is reproducible from config and saved artifacts.

**Restrictions:**

- Do not allow windows from the same raw file to appear in both train and test.
- Do not silently change window length, stride, channels, resampling rate, filtering, or normalization behavior.
- Do not tune preprocessing based on the test set; use training/validation only and report the final test once.
- Do not switch the main project to the Spadini sound-only/MFCC/XGBoost pipeline. Its preprocessing ideas may inspire optional experiments, but the main project remains MAFAULDA multichannel time-series windows and a lightweight Transformer for FPGA.
- Do not copy the Koutoulas full nine-model preprocessing/comparison pipeline. Use only relevant preprocessing ideas that support this project.
- Do not use synthetic data as the main evaluation source.
- If synthetic data is used for debugging or training augmentation, clearly mark it as debug-only or train-only augmentation.
- Any preprocessing added here must remain feasible for the later hardware path or must be clearly labeled as software-only analysis.

---

### Stage 3 — Software baseline models

**Status:** COMPLETE

**Goal:** Establish a software baseline for verifying that the data pipeline and labels are learnable.

**Expected code scope:**

- simple baseline model such as MLP or other lightweight classifier;
- training loop;
- validation metrics;
- test evaluation.

**Definition of done:**

- The baseline can train without crashing.
- Metrics are saved.
- Confusion matrix or per-class behavior can be inspected.
- Baseline is used only as a sanity check, not the final project architecture.

**Restrictions:**

- Do not let the baseline become the final project model.
- Do not add many non-Transformer baselines unless the user asks.

---

### Stage 4 — Lightweight Transformer baseline in Python

**Status:** COMPLETE

**Goal:** Implement and train the project’s main model family: a lightweight Transformer classifier suitable for later FPGA mapping.

**Expected code scope:**

- Transformer encoder model;
- small number of layers;
- limited hidden dimension;
- small number of attention heads;
- classification head using simple pooling such as mean/global average pooling;
- training and evaluation scripts;
- checkpoint and metrics export.

**Definition of done:**

- Transformer model trains on MAFAULDA windows.
- Validation/test metrics are generated.
- Best checkpoint is saved.
- Evaluation script can reload the checkpoint.
- Output dimensions match the approved class map.

**Restrictions:**

- Do not increase model size without checking FPGA feasibility.
- Do not introduce architecture features that are hard to quantize unless explicitly justified.
- Do not replace the Transformer with CNN/LSTM/GRU as the main architecture.

---

### Stage 5 — Model cleanup, reproducibility, and final software evaluation

**Status:** COMPLETE

**Goal:** Make the Python model path reproducible and report-ready before hardware conversion.

**Expected code scope:**

- deterministic seed handling;
- command-line or config-driven experiment settings;
- saved model config beside checkpoint;
- metrics JSON;
- confusion matrix generation;
- clear train/validation/test output;
- optional plot generation for report.

**Definition of done:**

- A clean run can reproduce the final software metrics.
- Model hyperparameters are saved.
- The exact checkpoint used for hardware conversion is identified.
- The evaluation includes accuracy, precision, recall, F1-score, and confusion matrix.
- ROC/AUC may be added if useful and not disruptive.

**Restrictions:**

- Do not tune using the test set.
- Do not overwrite previous best results without versioned artifact names.
- Do not change the approved data split while trying to improve metrics.

---

### Stage 6 — Quantization and fixed-point preparation

**Status:** COMPLETE (integer-only INT10 model; `docs/reports/consolidated_results_20261022.md`)

**Goal:** Convert the selected Transformer into a hardware-oriented numeric representation while measuring accuracy impact.

**Expected code scope:**

- quantization-aware or post-training quantization utilities;
- integer/fixed-point simulation in Python;
- weight export;
- scale/zero-point or fixed-point-format documentation;
- accuracy comparison between float model and quantized/fixed-point model.

**Definition of done:**

- The selected model has a fixed quantization format, preferably 8-bit or 16-bit depending on accuracy/resource tradeoff.
- Quantized inference can run in Python.
- Quantized accuracy drop is measured and reported.
- Weights/biases/scales are exported in a format usable by RTL/HLS.
- Golden input/output vectors are generated.

**Restrictions:**

- Do not quantize randomly without a repeatable export process.
- Do not change model architecture during quantization unless explicitly approved.
- Do not call a model hardware-ready until fixed-point behavior has been compared with the Python reference.

---

### Stage 7 — Hardware specification and golden vectors

**Status:** COMPLETE

**Goal:** Define the exact hardware-facing interface and generate test data for validating FPGA inference.

**Expected code scope:**

- input window serialization format;
- output class/confidence format;
- memory layout for weights;
- valid/ready or FIFO interface contract;
- golden vectors from Python quantized inference;
- testbench input/output files.

**Definition of done:**

- Every tensor dimension used by hardware is documented.
- The ordering of samples, channels, tokens, and features is fixed.
- The output class IDs match the Python class map.
- Golden vectors can be loaded by a simulator/testbench.

**Restrictions:**

- Do not invent a new tensor layout in RTL without matching Python export code.
- Do not modify golden vectors manually.
- Do not treat floating-point software outputs as hardware outputs after fixed-point is selected.

---

### Stage 8 — FPGA inference core

**Status:** COMPLETE — physically validated on the MYIR Z-Turn XC7Z020; the class is decided in FPGA logic

**Goal:** Implement the selected quantized Transformer inference path on FPGA or in a hardware-development flow targeting FPGA.

**Expected code scope:**

- input buffer/FIFO;
- token/window feeding logic;
- fixed-point arithmetic blocks;
- linear layers and activation approximations required by the selected model;
- attention/encoder computation for the approved lightweight architecture;
- pooling/classification head;
- class decision logic;
- simulation/testbench.

**Definition of done:**

- Hardware simulation matches golden vectors within the approved tolerance.
- Resource usage is reported: LUT, FF, BRAM, DSP.
- Latency and throughput are measured or estimated.
- The inference core produces one class result per input window.

**Restrictions:**

- Do not change the approved model to make RTL easier unless the user explicitly approves the architecture change.
- Do not remove attention and still call the result a Transformer.
- Do not ignore overflow/underflow behavior.
- Do not leave scaling constants undocumented.

---

### Stage 9 — Sensor or laboratory input path

**Status:** TODO — on-board preprocessing (raw samples -> per-window z-score -> INT10 on the FPGA) planned for Nov 2026; a real sensor stays optional

**Goal:** Add a real or laboratory input source when hardware permits, while keeping the prepared-window path usable.

**Expected code scope:**

- accelerometer or similar sensor interface, for example SPI or I2C;
- sampling-rate selection;
- streaming into an internal buffer;
- window assembly;
- overflow/underflow protection;
- optional debug capture to PC.

**Definition of done:**

- Sensor samples can be read reliably.
- Samples are windowed in the same format expected by the inference core.
- Buffer status is observable during debugging.
- Sampling rate and windowing behavior are documented.

**Restrictions:**

- Do not make the entire project fail if sensor hardware is unavailable.
- Do not change the model input format just for the sensor path.
- Do not mix uncalibrated real-sensor data into MAFAULDA evaluation metrics.

---

### Stage 10 — FPGA-to-PC communication

**Status:** COMPLETE over Ethernet (SSH/TCP); the UART result path on the board is planned for Nov 2026

**Goal:** Send classification results from the FPGA to the computer in a stable and debuggable format.

**Expected code scope:**

- UART/USB-to-UART transmitter path as the default simple option;
- optional binary protocol only after text protocol is stable;
- PC-side serial receiver;
- frame parsing and logging.

**Suggested initial message format:**

```text
FRAME=<id>,CLASS=<class_id>,LABEL=<label>,CONF=<confidence>\n
```

If confidence is not available in hardware, use a documented placeholder or send only the class ID/label.

**Definition of done:**

- PC receives one parseable message per classified window.
- Dropped/malformed frames are detected.
- The communication path is documented.

**Restrictions:**

- Do not design a complex protocol before the basic text protocol works.
- Do not change class IDs between Python, FPGA, and PC receiver.

---

### Stage 11 — PC-side user interface

**Status:** COMPLETE (Replay/Live viewer; a UI refresh is planned before the demo video)

**Goal:** Display FPGA classification results in a simple, presentable interface.

**Expected code scope:**

- serial receiver integration;
- live class label display;
- optional confidence display;
- optional log table;
- optional simple plot of recent predictions.

**Definition of done:**

- The UI shows the latest class result.
- It can run during a live demo.
- It can save or display a short log of predictions.

**Restrictions:**

- Do not overbuild the UI.
- Do not move classification logic into the UI.
- The UI should present results, not redefine them.

---

### Stage 12 — Final integration, validation, and report support

**Status:** COMPLETE for prepared windows (scope limits in `docs/project_status.md`)

**Goal:** Connect the software, quantization, hardware inference, communication, and UI work into a defensible final project.

**Expected code scope:**

- final run scripts;
- final metrics export;
- hardware resource/latency report collection;
- demo instructions;
- report figures/tables generated from artifacts;
- final documentation.

**Definition of done:**

- Software model metrics are documented.
- Quantized model metrics are documented.
- Hardware resource usage is documented.
- End-to-end data/result path is demonstrated.
- PC output display is demonstrated.
- Limitations are clearly stated.

**Restrictions:**

- Do not hide failed hardware or quantization results.
- Do not claim real-time behavior without latency/throughput evidence.
- Do not claim sensor generalization if only MAFAULDA prepared windows were evaluated.

---

## Recommended Development Workflow for Future Updates

For every future coding task:

1. Inspect the existing files first.
2. Identify which project stage the requested change belongs to.
3. Preserve existing interfaces unless the user explicitly asks for a breaking change.
4. Make the smallest safe change.
5. Add or update a runnable command.
6. Save outputs in `artifacts/` or the existing artifact convention.
7. Document any changed constants, tensor shapes, or file formats.
8. Run the relevant smoke test or explain exactly why it was not run.

---

## Canonical Experiment Rules

### Dataset and split

- Use MAFAULDA as the main dataset.
- Preserve file-level splitting to avoid leakage.
- Keep class distribution reporting available.
- Keep raw-data and generated-window artifacts separate.

### Model

- Main model family: lightweight Transformer encoder classifier.
- Use 1 or 2 encoder layers unless explicitly changed.
- Use a small number of heads.
- Keep hidden dimensions small enough for FPGA feasibility.
- Use simple pooling for the classification head when possible.

### Metrics

Minimum software metrics:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

Hardware/system metrics:

- LUT;
- FF;
- BRAM;
- DSP;
- latency;
- throughput;
- communication stability.

---

## Artifact and File Naming Guidance

Use clear names and avoid overwriting important results.

Suggested artifact names:

```text
artifacts/checkpoints/transformer_baseline_best.pt
artifacts/results/transformer_baseline_metrics.json
artifacts/results/transformer_baseline_history.json
artifacts/results/transformer_evaluation.json
artifacts/results/confusion_matrix.png
artifacts/quantized/transformer_quantized_config.json
artifacts/quantized/transformer_quantized_weights.*
artifacts/golden/golden_inputs.*
artifacts/golden/golden_outputs.*
```

If an existing naming scheme already exists, follow it instead of creating a parallel scheme.

---

## Commands to Verify Before Use

The exact module names may vary. Verify the repository before running. Previously used or expected commands include:

```bash
python -m src.data.check_data_quality
python -m src.train_transformer_baseline
python -m src.evaluate.evaluate_model
```

Some older context also mentioned:

```bash
python -m src.train.train_model
```

Do not assume both training paths are current. Inspect the repository and choose the maintained path.

---

## What Counts as a Bad Update

A code update is considered bad if it:

- improves accuracy by introducing data leakage;
- changes class IDs without updating all dependent code;
- breaks old evaluation scripts without a migration plan;
- makes the model too large for FPGA deployment;
- turns the project into a pure software study;
- moves final classification from FPGA to PC;
- uses extra papers as implementation requirements instead of background;
- modifies simulator/testbench behavior instead of fixing the implementation;
- hides assumptions about quantization, scaling, or tensor layout.

---

## Final Project Narrative to Preserve

This project demonstrates a complete engineering path for real-time machinery condition monitoring:

```text
MAFAULDA / prepared vibration windows
    -> preprocessing and windowing
    -> lightweight Transformer training in Python
    -> evaluation on unseen files
    -> model simplification and quantization
    -> FPGA inference path
    -> result transfer to PC
    -> simple user interface display
```

Every future update should support this narrative. Extra experiments are acceptable only if they do not disrupt the main path.
