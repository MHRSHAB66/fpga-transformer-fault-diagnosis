# Change Log — Advisor Deck

## 20-slide revision (structure + slide 7 redesign)
- **Removed** the standalone "Project Goal and Core Constraint" slide; its essential
  content was **merged into "End-to-End System Pipeline"** (now slide 3) as a subtitle
  ("4-class fault diagnosis with final inference on FPGA") plus a subtle output-class
  line (Normal · Imbalance · Misalignment · Bearing Fault). Deck is now **20 slides**;
  footers renumbered 2/20 … 20/20; the merged Persian speaker note replaces the two
  old notes.
- **Redesigned slide 7 → "Frequency Analysis and Sampling-Rate Decision"**: a native
  3-part layout — (A) prior-work / alternative downsampling path, (B) a native bar
  chart of the acc_axial_b1 22–23 kHz energy fraction by class (Normal 86.6 · Imbalance
  83.2 · Misalign 82.7 · Bearing 3.2 %) as the visual focus, (C) a decision card — plus
  a bold banner: "Keep 50 kHz as the current baseline; evaluate downsampling only via
  controlled ablation." Numbers from
  `artifacts/results/frequency_investigation/investigation_summary.json`. The old
  framed frequency-spectrum image is no longer used on this slide (now fully native).
- **Fixed** the `build_native_deck.py` slide-removal loop that had produced a duplicate
  "Today's Meeting" slide (22-slide franken state): sldId entries are now removed first
  (with an assertion), then the orphan relationships are dropped.
- Revert base: `FPGA_Fault_Diagnosis_Progress.IMG21.pptx` (clean pre-native deck with
  the hand-made title slide; the native deck is regenerated on top of it).

---

# Change Log — 21-Slide Advisor Deck (superseded)

File: `docs/slide_handoff/FPGA_Fault_Diagnosis_Progress.pptx`
Revert points:
- `FPGA_Fault_Diagnosis_Progress.IMG21.pptx` — previous image-per-slide 21-slide deck
  (revert point for the native-shapes rebuild).
- `FPGA_Fault_Diagnosis_Progress.PRE21.pptx` — older 17-slide version.
Build scripts: `src/tools/build_native_deck.py` (current), `src/tools/add_notes.py`.
Superseded image build: `src/tools/build_figures.py`, `src/tools/deck_figures.py`,
`src/tools/build_deck.py`.

## Native rebuild (editability pass)
The whole deck was rebuilt with **native, editable PowerPoint objects** instead of
one flattened figure per slide. Every card, process box, roadmap stage, decision
box, KPI card, label chip, progress bar, comparison table, block-diagram component
and callout is now a real shape / table / chart:
- **Shapes:** rounded rectangles, connectors with arrowheads, accent bars, chips.
- **2 native tables:** tuning metric comparison (slide 10) + confusion matrix
  (slide 12, heat-shaded cells).
- **5 native charts:** tuning parameter bar (10), training line chart (11),
  per-class recall/F2 grouped bar (12), quantization bar (14), memory bar (16).
- **Only remaining image (besides logos):** the measured frequency-spectrum plot
  (slide 8), placed inside a native rounded frame + gold callout.
Deck-wide additions: a consistent **dark charcoal 1.75 pt outline** around every
block/card, and a **deep-purple underline** directly under every content-slide title.
Preserved: title slide (index 0), footer style + numbering, palette, slide order,
project narrative, and all 21 Persian speaker notes.

## Preserved
- **Slide 1 (title):** untouched — the hand-edited title slide (English AUT seal,
  growth icon, project title, subtitle, student + advisor names).
- **Footer style:** gold rule + AUT logo bottom-left + "Machinery Fault Diagnosis
  on FPGA" + page number, applied consistently to slides 2–21 (auto-numbered N/21).
- **Palette / identity:** white background; purple #281538, blue #83B6E3,
  gold #F1C232, grey #999999; Montserrat titles.

## Structure (title + 20 content slides)
2 Today's Meeting · 3 Project Goal & Core Constraint · 4 End-to-End System Pipeline ·
5 Progress at a Glance · 6 MAFAULDA Dataset · 7 Leakage-Free Split & Windowing ·
8 Frequency Analysis · 9 Selected Software Baseline Architecture ·
10 Model Tuning & Configuration Selection · 11 Final A_baseline Training Run ·
12 Held-Out Test Performance · 13 Safety-Oriented Evaluation · 14 Quantization Results ·
15 Fixed-Point Reference & Golden Vectors · 16 Hardware Feasibility & Bottlenecks ·
17 Proposed FPGA Inference Architecture · 18 Key Design Decisions Before HLS ·
19 Decisions Requested from the Advisor · 20 Next Milestones · 21 Progress Summary.

## Added (new slides)
Today's Meeting (agenda + roadmap), Progress at a Glance, Model Tuning &
Configuration Selection, Proposed FPGA Inference Architecture, Key Design
Decisions matrix, Decisions Requested cards, Next Milestones timeline.

## Redesigned / merged
- Pipeline redrawn as a real end-to-end flow (financial Slidesgo icons removed).
- Dataset + split/windowing separated into a dataset-infographic slide and a
  leakage-free split+windowing slide.
- Quantization, golden vectors, memory, safety, test redrawn with cleaner charts.

## Charts created (figures_v2/)
agenda, goal, system_pipeline, progress_glance, dataset_full, split_windowing,
frequency_evidence, architecture_diagram, tuning_comparison, training_behavior,
test_results, safety, quant_results, golden_flow, hw_feasibility,
fpga_architecture, decision_matrix, decisions_cards, milestones, summary_status.

## External assets
- Amirkabir University of Technology seal (title + footer). Growth icon (title).
- No fabricated logos. Vitis/AMD logos NOT added (kept text-only) to avoid
  implying a synthesized toolchain result.

## Data-context labeling (resolved the "which number is which" issue)
- **98.22%** — held-out test set (final complete A_baseline run), 95,404 windows.
- **98.19% / 98.28% / 97.46%** — validation-set tuning (max 12 epochs), labeled as such.
- **~98.14%** — quantization comparison run (29,280-window subset).
- **98.46%** — fixed-point INT16 evaluated subset.
All four appear only with their evaluation context. Tuning explicitly labeled
"best overall among the three completed runs" (not a global optimum);
E_3layers/C_d48/D_d64_slow marked "planned but not executed".

## Honesty guardrails
No FPGA/HLS/synthesis/LUT/FF/BRAM/DSP/latency/throughput/UART numbers invented.
Memory figures labeled "analytical estimates — not post-synthesis". FPGA
architecture labeled "proposed — not yet synthesized". Golden vectors labeled
"sanity check; full proof requires HLS co-simulation". Real-sensor question omitted.

## Tuning source of truth
All tuning values read from
`artifacts/runs/tuning_20260616_161108/tuning_final_report.json`.
