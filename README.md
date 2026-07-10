# SENTRY + PSID-8

Reference implementation of the two artifacts of the paper:

- **PSID-8 toolkit** — everything needed to build the benchmark: compositional
  annotation schema (imported verbatim from the CVAT label config piloted on
  images), annotation guide, scenario-stratified camera-disjoint splits,
  inter-annotator agreement, statistics, and integrity checks.
- **SENTRY** — single-stage detector with a Temporal Feature Memory (ConvGRU per
  FPN level) + motion-gated attention, temporal-consistency loss, tube linking,
  event confirmation, and structured alert records.

## Layout
```
psid8/            benchmark toolkit (numpy/sklearn - runs anywhere)
sentry/           model, losses, data, tubes, metrics (PyTorch for the model;
                  tubes and metrics are pure numpy)
configs/          base config (hyperparameters = search space, not fixed values)
protocol/         anti-manipulation pre-registration
notebooks/        Kaggle workflow (2x T4)
tests/            unit tests for the offline-runnable parts
```

## Running on Kaggle
Option A (recommended): upload this repository zip as a Kaggle Dataset named
`sentry-psid8`; Kaggle auto-extracts it. In the notebook:
```python
import sys; sys.path.insert(0, "/kaggle/input/sentry-psid8/sentry-psid8")
```
Option B: `!git clone <your-repo-url> /kaggle/working/repo` then insert that path.
Option C: `pip install -e /kaggle/working/repo` (pyproject.toml included).

## Integrity rule
No number enters the paper unless it comes from logs produced by this
repository. The test split is frozen in Phase 1 and evaluated ONCE per final model.

## Como citar

Se este software ou o conjunto de ferramentas for útil ao seu trabalho, cite tanto
o artigo quanto o software arquivado.

**Artigo:**
```bibtex
@article{[chave]2026sentry,
  title   = {Rumo à Detecção de Incidentes de Segurança em Vídeo: a Arquitetura SENTRY e a Lacuna de Dados},
  author  = {[Sobrenome], [Nome]},
  journal = {[Periódico ou conferência após aceitação]},
  year    = {2026},
  doi     = {[DOI do artigo]}
}
```

**Software (DOI do Zenodo, versão arquivada):**

```bibtex
@software{[chave]2026sentry_sw,
  title     = {SENTRY + PSID-8: detecção espaço-temporal de incidentes e conjunto de ferramentas de benchmark},
  author    = {[Sobrenome], [Nome]},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.[A_SER_EMITIDO]},
  url       = {https://github.com/[USUARIO]/sentry-psid8}
}
```

O botão "Cite this repository" do GitHub lê o arquivo `CITATION.cff` e gera as
citações automaticamente em APA e BibTeX.

## Licença

Código sob licença MIT (ver `LICENSE`). O conjunto de dados de vídeo PSID-8,
quando publicado, será distribuído sob licença de dados própria (CC BY 4.0),
em depósito separado.
