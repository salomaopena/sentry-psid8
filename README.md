# SENTRY + PSID-8

Reference implementation of the two artifacts of the paper *"Towards Physical
Security Incident Detection in Video Surveillance: The SENTRY Architecture and
the Data Gap"*:

- **PSID-8 toolkit** — everything needed to build the benchmark: the
  compositional annotation schema (imported verbatim from the CVAT label
  configuration piloted on images), annotation guide, scenario-stratified
  camera-disjoint splits, inter-annotator agreement, statistics, and integrity
  checks.
- **SENTRY** — a single-stage detector with a Temporal Feature Memory (ConvGRU
  per FPN level) and motion-conditioned attention, a temporal-consistency loss,
  tube linking, event confirmation, and structured alert records.

## Layout

```text
psid8/            benchmark toolkit (numpy/sklearn - runs anywhere)
sentry/           model, losses, data, tubes, metrics (PyTorch for the model;
                  tubes and metrics are pure numpy)
sentryc           physical-cyber convergence layer. 
configs/          base configuration (hyperparameters = search space)
protocol/         anti-manipulation pre-registration
notebooks/        Kaggle workflow (2x T4)
tests/            unit tests for the offline-runnable parts
```

## Running on Kaggle

Upload this repository as a Kaggle Dataset named `sentry-psid8`; Kaggle
auto-extracts it. In the notebook:

```python
import sys; sys.path.insert(0, "/kaggle/input/sentry-psid8/sentry-psid8")
```

Alternatives: `!git clone https://github.com/salomaopena/sentry-psid8.git` or `pip install -e` (pyproject.toml
included).

## Integrity rule

No number enters the paper unless it comes from logs produced by this
repository. The test split is frozen in Phase 1 and evaluated once per final
model (see `protocol/PREREGISTRATION.md`).

## How to cite

If this software or toolkit is useful to your work, please cite both the paper
and the archived software.

**Paper:**

```bibtex
@article{[key]2026sentry,
  title   = {Towards Physical Security Incident Detection in Video Surveillance:
             The SENTRY Architecture and the Data Gap},
  author  = {Pena, Salomão Bento Nilo; Souza, Jefferson R.; Nomura, Shigueo},
  journal = {[Journal or conference after acceptance]},
  year    = {2026},
  doi     = {[Paper DOI]}
}
```

**Software (Zenodo DOI, archived version):**

```bibtex
@software{[key]2026sentry_sw,
  title     = {SENTRY + PSID-8: spatiotemporal incident detection and benchmark toolkit},
  author    = {Pena, Salomão Bento Nilo; Souza, Jefferson R.; Nomura, Shigueo},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.[TO_BE_MINTED]},
  url       = {https://github.com/salomaopena/sentry-psid8}
}
```

GitHub's "Cite this repository" button reads `CITATION.cff` and generates APA
and BibTeX citations automatically.

## License

Code released under the MIT License (see `LICENSE`). The PSID-8 video dataset,
once published, will be distributed under a separate data license (CC BY 4.0)
in its own deposit.

## Kappa

```python
python psid8/scripts/agreement.py anotadorA.json anotadorB.json
```
