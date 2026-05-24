# Portfolio Ship Plan — gps-denied-navigation

**Status:** active (compressed timeline as of 2026-05-22)
**Owner:** Ishaan; Claude has §1 + §6 override for this project only — full code execution authorized
**Target date:** 2026-05-26 (Tue) — apply same week
**Audience:** recruiters reviewing for Summer 2027 ML/robotics internships (PI, Figure, Skild, DeepMind Robotics, Tesla Optimus, Nvidia GEAR)

## REVISION 2026-05-22: 4-day compressed timeline

Original 3-week plan is too long given the application deadline. Compressed plan:

| Day | Date | Focus | Deliverables |
|---|---|---|---|
| 1 | Fri 5-22 | Foundation | Decision docs 027 + 028; unified eval harness; hero figure script; loss-curves plot |
| 2 | Sat 5-23 | Headline experiment | RLS adaptation head on top of v15 (lowest-risk adaptation approach); train, eval, add to results table |
| 3 | Sun 5-24 | Visuals + repo cleanup | Animated GIF for README; plot polish; decision doc 029 (RLS findings); dead-code pass |
| 4 | Mon 5-25 | README + distribution | README rewrite, Notion writeup, demo video, repo polish, LinkedIn announcement |
| Apply | Tue 5-26 | — | — |

Original 3-week structure below is **superseded** — kept for reference.

---

## ORIGINAL 3-WEEK PLAN (superseded)

---

## 1. Pivot rationale

The project has crossed the "real result" bar. LSTM v15 + velocity-only filter beats every prior baseline at 30-second GPS outages (0.405 m/s final-position error, ~4× the GPS-aided EKF oracle of 0.104). Decision doc 022 already establishes that the remaining gap is structural distribution shift, not solvable by squeezing more per-step accuracy out of the IMU model.

Three more experimental directions were on the table (sequence-level adaptation, val_final checkpoint-selection retrain, longer-outage curriculum). Each has high implementation cost relative to the marginal headline gain — and none of them improve the project's *visibility*, which is the actual bottleneck for the Summer 2027 cohort.

**The lever now is packaging, not modeling.** Recruiter time on a portfolio piece is ~30 seconds. A compelling hero figure + one-paragraph hook + 5-minute reproducibility path will outperform a 0.05 m/s improvement on a metric the recruiter won't read.

## 2. Success criteria

A reviewer landing on the repo's GitHub page should, within 60 seconds:
- See a hero figure that visually communicates the problem and the result
- Read a one-sentence hook in the README
- See the headline numbers vs. baselines
- Find a one-command reproducibility path

A reviewer with 5 minutes should be able to:
- Read a technical writeup that explains the design choices
- Run the eval script and reproduce the numbers
- Watch a demo (video or animated GIF)

A reviewer with 30 minutes should be able to:
- Read the chronological decision log (decisions 001–028)
- Inspect the code and understand the architecture

## 3. Three-week timeline

| Week | Deliverable | Owner | Output |
|---|---|---|---|
| 1 (5-22 → 5-29) | Hero figure | Claude | `scripts/make_hero_figure.py` → `docs/figures/hero.png` (and `.svg`) |
| 1 | README rewrite | Claude (draft) + Ishaan (voice) | `README.md` — hook, hero image, results table, quickstart, links to docs |
| 1 | Quickstart eval | Claude | `scripts/quickstart_eval.py` — downloads a small EuRoC slice, runs v15, prints metrics + plots |
| 2 (5-29 → 6-5) | Technical writeup | Claude (draft from decision docs) + Ishaan (voice) | `docs/writeup.md` — narrative through the project; also publish as Notion blog post |
| 2 | Demo video / GIF | Claude (storyboard, rendering) + Ishaan (voiceover if video) | `docs/figures/demo.gif` or `docs/demo.mp4` — visualizes 30s outage with and without nav model |
| 2 | (Background, no critical path) v15 retrained on val_final | Ishaan (one-line edit to `train_lstm_v15.py`) → desktop training | If it lands by week 3, swap into hero figure; else punt |
| 3 (6-5 → 6-12) | Polish + ship | Claude | Decision docs 027, 028; CHANGELOG; badges; cross-links; deploy writeup to Notion + repo Pages |

## 4. Deliverable details

### 4.1 Hero figure (week 1, day 1)
- Test sequence: `MH_05_difficult` (test set, never seen by model)
- Pick a 60-second window with non-trivial motion (curves, altitude change)
- Simulate a 30s GPS outage in the middle
- Plot three trajectories:
  - Ground truth (Leica)
  - IMU-only dead reckoning (the bad baseline)
  - v15 + velocity-only filter (our solution)
- Below: position-error-vs-time chart showing the three approaches diverging
- Output: `docs/figures/hero.png` (1600×900, web-ready) + `.svg` (vector)
- Script: `scripts/make_hero_figure.py` — reproducible, runs from clean clone

### 4.2 README rewrite
- One-sentence hook above the fold
- Hero figure embedded
- Results table (3 rows: IMU-only / TCN v7 / LSTM v15+filter, 1 column: 30s final-position error)
- Quickstart (3 commands max)
- Architecture diagram (existing or new)
- Links to: decision docs index, technical writeup, dataset, model spec
- Badges: Python version, PyTorch version, license

### 4.3 Quickstart eval
- `scripts/quickstart_eval.py --sequence MH_05_difficult`
- Downloads (or assumes) a single EuRoC sequence
- Loads `checkpoints/lstm_v15.pt` and `data/processed/splits/normalization_stats.json`
- Runs the nav eval harness for a single 30s outage
- Prints metrics + saves a plot to `docs/figures/quickstart_result.png`
- Total runtime: under 1 minute on CPU

### 4.4 Technical writeup
- Source material: 27 existing decision docs + CLAUDE.md results table
- Structure: problem → data → baseline → instrumentation → key decision points (in chrono order, summarized) → final architecture → results → limitations → what we'd do next
- Length: ~1500 words, ~6 figures
- Output: `docs/writeup.md` + Notion page under `Projects/GPS-Denied Nav`

### 4.5 Demo video / GIF
- 30-second animated GIF showing the hero-figure trajectories building up in real time
- Or: 60-second MP4 with voiceover walking through the same content
- Decision deferred to week 2 based on bandwidth

## 5. Deferred experiments

These are *not* killed — they're queued for after Summer 2027 application close (Nov 2026). Each gets a decision doc explaining the defer rationale so future-me knows where to pick up.

- **(a) Sequence-level real-time adaptation** (TTT / LoRA / RLS head) — high implementation cost, high research uncertainty. Best case ~2× improvement on the structural gap; worst case ~0. Deferred.
- **(b) v15 retrained selecting on val_final** — cheap (one-line change, one training run). Kept as a background task; if it lands before week 3, the better number goes in the README.
- **(c) Longer-outage curriculum / bigger model** — moderate cost, unclear reward. Deferred.

## 6. Risks

- **Hero figure is unconvincing.** Mitigation: try 3 different test windows, pick the most visually compelling. If still flat, switch to a side-by-side animation rather than static trajectories.
- **Notion/GitHub Pages publishing friction.** Mitigation: ship to GitHub Pages first (zero auth), Notion is bonus.
- **Recruiters never see this.** Mitigation: link from resume, LinkedIn post on completion, ping on Twitter/X with hero figure. Distribution > polish.
- **Background v15-on-val_final retrain produces a much better number after week 3.** Mitigation: cheap to swap into README post-hoc; nothing depends on the exact number.

## 7. Decision docs to write during the ship

- **027** — v15/v16 loss function exploration findings (already deferred from session 2026-05-21)
- **028** — Pivot from more experiments to portfolio ship (this doc summarizes the *what*; 028 captures the *why* in decision-doc format)

## 8. After ship

Once shipped and resume is updated:
- Return to roadmap Phase 0/1 (DL Fundamentals) per `~/projects/CLAUDE.md`
- Sequence-level adaptation experiment becomes a candidate for Phase 3 P1
