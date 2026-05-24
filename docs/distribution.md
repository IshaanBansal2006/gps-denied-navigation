# Distribution

Drafts and assets for getting the project in front of recruiters. Paste-ready,
edit-as-needed.

---

## LinkedIn post (short)

> 🛰️ Shipped: a neural-aided IMU navigator that keeps a drone within **3 m of
> ground truth after 30 seconds of GPS denial** on EuRoC MH_05 — 2.5× the
> GPS-aided EKF oracle, 170× better than naive IMU dead-reckoning.
>
> Built end-to-end: data pipeline → 16 model variants (TCN → LSTM v15) → 
> velocity-only Kalman filter → RLS-adapted final head that specializes online
> to the current sequence. 29 decision docs along the way.
>
> Headline result + honest limitations + reproducible quickstart in the repo:
> https://github.com/IshaanBansal2006/gps-denied-navigation

---

## LinkedIn post (long — for posting the day after if traction is low)

> When GPS dies on a drone — jammers, indoor flights, contested airspace — naive
> IMU integration drifts ~620 m in 30 seconds. That's a lost vehicle.
>
> Over the past few months I've been building a neural-aided IMU navigator from
> scratch. The system:
> - Trains an LSTM to predict per-step velocity from raw 6-DoF IMU (gyro + accel)
> - Wraps it in a velocity-only Kalman filter that I derived from a 15-state EKF
> - **Adapts a final linear head online** using pre-outage GPS-aided velocity (RLS)
>
> Final result on EuRoC MH_05_difficult (held-out test sequence):
> **0.259 m/s final velocity error after 30 seconds of simulated GPS denial.**
> That's 2.5× the GPS-aided EKF oracle (the theoretical ceiling) — and 170×
> better than naive IMU dead-reckoning.
>
> Most interesting moments:
> - Switching from Δv to absolute-velocity targets (decision 017) was the single
>   biggest unlock. Going from "this model doesn't work" to "this model has
>   positive R²" overnight.
> - Strapdown EKF during GPS outage is *harmful* — attitude drift poisons IMU
>   propagation within 10s. A velocity-only filter (no attitude, no position
>   state) outperformed the 15-state EKF (decision 019).
> - Yaw-rotation augmentation collapsed performance 20× — turns out EuRoC's
>   heading priors are load-bearing (decision 020).
> - The RLS adaptation head only helps at the 30-second horizon the model was
>   trained for. Tries to be a silver bullet and ends up specialized to one
>   scenario (decision 029).
>
> 29 decision docs, 16 model variants, full reproducibility.
>
> https://github.com/IshaanBansal2006/gps-denied-navigation

---

## Twitter / X post

> built a neural IMU navigator that keeps a drone within 3 m of truth after
> 30 s of GPS denial on EuRoC — 2.5× the GPS-aided oracle, 170× better than
> dead-reckoning. 16 model variants + 29 decision docs, all reproducible.
> https://github.com/IshaanBansal2006/gps-denied-navigation

---

## Resume bullet (one-line)

> Built a neural-aided IMU navigator (PyTorch LSTM + Kalman filter + online RLS
> adaptation) achieving 0.259 m/s velocity error after 30 s of GPS denial on
> EuRoC MH_05_difficult — 2.5× the GPS-aided EKF oracle. 16 model variants,
> 29 decision docs, full reproducibility.

## Resume bullet (two-line)

> **GPS-denied UAV navigation** — built a neural-aided IMU navigator from
> scratch in PyTorch: dataset pipeline, LSTM regressor trained on 6 EuRoC
> sequences with end-to-end navigation loss, velocity-only Kalman filter, and
> online RLS adaptation of the final head. Final result: 0.259 m/s velocity
> error after 30 s of GPS denial on the held-out test sequence — 2.5× the
> GPS-aided EKF oracle, 170× better than dead-reckoning. 16 model variants
> documented in 29 decision docs.

---

## Email pitch (for cold outreach)

> Subject: GPS-denied UAV navigation — 30s nav error 2.5× the GPS oracle
>
> Hi [name],
>
> I've spent the last few months building a neural-aided IMU navigator for
> UAVs operating under GPS denial — the kind of system Shield.AI's Hivemind
> hints at. The headline result: 0.259 m/s final velocity error after a
> 30-second simulated GPS outage on the held-out EuRoC MH_05 sequence, which
> is 2.5× the GPS-aided EKF oracle and 170× better than naïve IMU
> dead-reckoning.
>
> The whole thing — pipeline, model, filter, evaluation — is open-source with
> 29 decision docs walking through every architectural choice (including the
> dead ends). I'd love to talk about it, if you have 15 minutes.
>
> https://github.com/IshaanBansal2006/gps-denied-navigation
>
> Best,
> Ishaan

---

## Where to post

| Channel | Why | When |
|---|---|---|
| LinkedIn (short post) | Recruiters scroll there. Hook + repo link. | Day-of ship |
| LinkedIn (long post) | If the short post gets <50 impressions in 24 h, post the long version with the storytelling angle | Day +1 |
| Twitter/X | Robotics + ML twitter is alive (Hugging Face, latent space, etc.) | Day-of |
| Hacker News (Show HN) | Worth a shot; if it lands the visibility is massive | Day +2 (off-peak; less competition) |
| Reddit r/robotics, r/MachineLearning | Niche audiences who care | Day +1, with a comment context |
| Direct cold outreach | Email engineers at Skild, PI, Figure, Anduril, Shield.AI | Within 1 week |

---

## Checklist before posting

- [ ] README hero image renders correctly on GitHub mobile + desktop
- [ ] Animated GIF plays inline in the rendered README
- [ ] All decision-doc links resolve (no 404s)
- [ ] `pip install -r requirements.txt && python3 scripts/neural_aided_ekf_lstm_v15_rls.py --outages 30` actually reproduces 0.259 m/s in a fresh clone (with EuRoC MH_05 downloaded)
- [ ] Pin the repo on your GitHub profile
- [ ] Update LinkedIn "Featured" section with the repo link
- [ ] Resume bullet updated and exported
