# Experimental protocol pre-registration (anti-manipulation)

**Hypothesis (falsifiable):** explicit short-window temporal reasoning inside a
single-stage detector improves detection on temporally defined classes (fall)
relative to a static-signature control class (fire), at a latency cost <= [X]%,
measured by the delta vs. the W=1 ablation and the frame-level baseline. If the
gains do not concentrate on the temporal class, the hypothesis will be reported
as refuted.

## Phases (mandatory order)

1. **Freeze**: integrity checks PASS -> commit registered -> hashes recorded.
   No script opens the test split until Phase 5.
2. **Stage A**: frame-level training (Ultralytics) - model selection by
   validation mAP@50.
3. **Stage B**: temporal training (TFM + L_tc) - hyperparameter search on the
   validation split ONLY; every trial logged (W&B/MLflow) and exported.
4. **Ablations**: -TFM, -motion gate, -L_tc, W in {1,4,8,16} - on validation.
5. **Test**: ONE evaluation per finalized model (SENTRY + each baseline).
   Seeds {0,1,2}; mean +/- std; bootstrap CI n=1000; McNemar + paired t-test.
6. **Reporting**: numbers copied from logs by script, never typed by hand.
   A convergence failure is reported as a failure, not as a number.

## Prohibitions

No peeking at the test split; no seed cherry-picking; no post-hoc removal of
hard examples; no new metrics chosen after seeing results.
