# Notas para a Tese — Sessão de Auditoria de Código (manter actualizado)

Este ficheiro existe para que nada do que foi discutido durante a auditoria de
engenharia se perca antes da escrita da tese. Acrescente-lhe conteúdo; não o
deixe desactualizado.

---

## 1. Alcance do impacto: a auditoria invalida os resultados do Artigo 1?

**Não.** Isto precisa de ficar bem claro e ser defendido se um revisor colocar
a mesma pergunta.

O bug crítico encontrado nesta auditoria (`sentry/train.py` a treinar
silenciosamente nada) vivia num **script autónomo (CLI)**. Os números
reportados no artigo (Tabela 2, ablação de épocas em 5/10/30) vieram de um
**caminho de código diferente, já a funcionar**: a célula de Estágio B do
notebook Kaggle (`[5]`), que tinha a sua própria cópia inline do loop de
treino, independente do `sentry/train.py`, e já usava a formulação
supervisionada correcta (`v8DetectionLoss` + `L_tc` auxiliar).

A prova de que esse loop inline treinou de verdade, e não só o raciocínio:
a perda de treino (`det`) caiu de forma **monótona** à medida que as épocas
aumentavam (16,7 → 14,1 → 12,7 a 5/10/30 épocas). Esse padrão não pode
ocorrer se `backward()`/`optimizer.step()` nunca tivessem corrido — é
exactamente a assinatura que o script autónomo com bug **não conseguiria**
produzir (nesse caso, a perda ficaria presa no valor inicial para sempre,
como reproduzido em `tests/test_train.py`).

**Conclusão:** o sobreajuste reportado (a perda de treino cai enquanto o F1
de teste degrada) é um achado real sobre escassez de dados, não um artefacto
de código com defeito. O diagnóstico do Artigo 1 mantém-se. O bug é um
defeito latente num ponto de entrada alternativo que nunca foi usado para
produzir um número publicado — importante corrigir para que ninguém tropece
nele mais tarde, mas não muda o que já foi reportado.

Um ponto mais estreito, também importante: a correcção do `coco_to_yolo.py`
(o layout `clips` vs. `flat`) também não afecta o Artigo 1. O pipeline do
Le2i (`le2i_to_clips.py`) escreve directamente para o layout canónico
`clips/<id>/frames+labels` sem nunca passar pelo `coco_to_yolo.py`. Essa
correcção importa para o **futuro** benchmark PSID-8 em vídeo (CVAT → COCO →
`coco_to_yolo.py`), não para nada já publicado.

---

## 2. Compatibilidade com versões do YOLO/Ultralytics — o que está de facto verificado

Dois eixos de variação independentes, e agora **ambos verificados ao vivo**.

**Eixo A — versão do pacote `ultralytics`** (ex.: 8.3.49 fixada no Kaggle vs.
8.4.103 disponível no sandbox de auditoria). Foi aqui que se encontrou uma
incompatibilidade real e verificada: a saída bruta do modelo em modo treino
mudou de forma entre estas versões (lista simples de features vs. um `dict`
`{"boxes","scores","feats"}`), e a forma esperada pelo `v8DetectionLoss`
mudou também. Corrigido: `sentry/train.py` agora entrega a saída bruta ao
critério **sem modificação** (cada versão sabe interpretar a sua própria
forma) e usa `extract_feat_list()` apenas para o termo auxiliar `L_tc`.
Coberto por
`tests/test_train.py::test_extract_feat_list_handles_both_ultralytics_shapes`.

**Eixo B — geração da arquitectura YOLO** (v8, v11, v26). **Agora totalmente
verificado ao vivo**, não apenas por leitura de documentação: descarreguei e
treinei de facto os três modelos (`yolov8n`, `yolo11n`, `yolo26n`) através do
mesmo código, confirmando gradientes reais (todos os 27 tensores da TFM
mudam após o treino em todos os três casos).

| Geração | Cabeça | DFL | Compatível com `build_criterion`? |
|---|---|---|---|
| YOLOv8 (`yolov8n`) | `Detect` ancora-livre, desacoplada | sim, reg_max=16 | **Verificado ao vivo** — `criterion=v8DetectionLoss` |
| YOLO11 (`yolo11n`) | mesma cabeça `Detect` da v8 (só mudam blocos do backbone/pescoço: C3k2 em vez de C2f) | sim, reg_max=16 | **Verificado ao vivo** — `criterion=v8DetectionLoss` |
| YOLO26 (`yolo26n`) | cabeça dupla `Detect` (one-to-one sem NMS + one-to-many), `end2end=True` | **removido** (reg_max=1) | **Verificado ao vivo** — `criterion=E2ELoss` |

**A solução que tornou isto possível:** em vez de construir manualmente
`v8DetectionLoss` com hiperparâmetros fixos, `build_criterion()` agora chama
`det_model.init_criterion()` — o próprio método que o Ultralytics expõe em
cada `DetectionModel` para escolher a perda certa (`E2ELoss` quando
`model.end2end` é verdadeiro, como no YOLO26; `v8DetectionLoss` caso
contrário). Isto delega a decisão a quem já a conhece de fábrica, em vez de
eu tentar adivinhá-la ou fixá-la a uma arquitectura específica. O
`extract_feat_list()` foi generalizado para reconhecer a terceira forma de
saída observada (o `dict` aninhado `{"one2many": {...}, "one2one": {...}}`
do YOLO26), usando por omissão o ramo `"one2one"` (o de inferência, sem NMS)
para o termo `L_tc`.

Coberto por
`tests/test_train.py::test_stage_b_trains_across_yolo_generations`, que
descarrega os três modelos, treina cada um por uma época em dados sintéticos,
e confirma que todos os parâmetros da TFM mudam nos três casos.

**Conclusão actualizada:** pode usar o YOLO26 como detector base para
facilitar a implantação em dispositivos de borda (a cabeça sem NMS reduz a
latência de pós-processamento). O código já trata isto correctamente, sem
necessitar de nenhum ramo específico para essa arquitectura.

---

## 3. Ficheiros alterados nesta auditoria (para fundir na cópia de trabalho)

**Modificados** (substituir estes ficheiros exactos):
`psid8/scripts/coco_to_yolo.py`, `psid8/scripts/integrity_check.py`,
`sentry/aggregate.py`, `sentry/train.py`, `sentry/data.py`, `pyproject.toml`,
`sentry/__init__.py` (só a versão), `sentryc/graph_builder.py`,
`sentryc/metrics.py`, `sentryc/gnn_correlation.py`, `sentry/plots.py`,
`psid8/scripts/dataset_stats.py`, `tests/run_tests.py`, `tests/test_sentryc.py`,
`ARCHITECTURE.md`, `CHANGELOG.md`, **`notebooks/sentry_kaggle.ipynb`** (célula
[5] reescrita — ver secção 6).

**Adicionados** (novos ficheiros): `tests/test_psid8_scripts.py`,
`tests/test_train.py`, `THESIS_NOTES.md` (este ficheiro).

**Intocados, confirmados por hash antes/depois**: `sentry/modules.py`,
`sentry/ultralytics_adapter.py`, `sentry/tubes.py`, `sentry/metrics.py`,
`sentry/seeds.py`, `sentry/stageb_train.py`,
`psid8/scripts/le2i_to_clips.py`, `psid8/scripts/build_splits.py`,
`psid8/scripts/agreement.py`, `psid8/scripts/curate_clips.py`,
`sentryc/network_stream.py`, `sentryc/alerts.py`.

**Instrução prática de fusão:** a forma mais segura é sobrepor a pasta de
trabalho local inteira com o conteúdo do zip entregue, e depois usar `git
diff` contra a cópia de trabalho antes de fazer commit, para que o conjunto
exacto de alterações fique visível e revisável em vez de fundido às cegas.

---

## 4. Experiência de tempo/memória: metodologia para reproduzir à escala real

Os números já medidos (2× menos chamadas a `model()`, 4,05× de redução no
tempo de relógio para uma época) foram obtidos com **dados de brinquedo num
sandbox CPU** (`yolov8n` minúsculo, imagens 64×64, 4 clipes sintéticos,
`batch_size=2`) — reais e reproduzíveis, mas **não são os números a citar no
artigo**. Servem só para provar que o mecanismo (menos chamadas → menos
tempo) é real, não apenas teórico.

**Para obter números citáveis**, repita a mesma comparação no Kaggle, à
escala real do artigo (clipes do Le2i, `imgsz=640`, `yolov8m`, o split de
treino real, `window=8`). Protocolo sugerido para a Secção V (ou uma nova
subsecção de implementação/custo computacional):

1. Correr uma época do Estágio B com o loop antigo, clipe a clipe (disponível
   no histórico do git / na lista de ficheiros desta secção) e registar o
   tempo de relógio + memória GPU de pico
   (`torch.cuda.max_memory_allocated()`).
2. Correr uma época do Estágio B com o **novo** loop em lote
   (`sentry.train.run_stage_b` + `collate_batched`), mesmos dados, mesma
   seed, mesmo `batch_size`. Registar os mesmos dois números.
3. Reportar: número exacto de chamadas ao modelo (pelo desenho: `T` vs.
   `N*T`), razão de tempo de relógio, razão de memória de pico. Declare o
   tamanho do lote usado, já que a redução de chamadas escala com ele.
4. Confirmar equivalência numérica: os dois loops devem produzir trajectórias
   de `det`/`tc` estatisticamente indistinguíveis ao longo de uma época
   (mesma fórmula de perda, mesmos rótulos, mesmo modelo) — reporte isto como
   verificação de sanidade, já que uma alegação de velocidade legítima exige
   mostrar que a correcção foi preservada.

`tests/test_train.py::test_batched_loop_uses_fewer_model_calls` é o modelo
para a instrumentação de contagem de chamadas dos passos 1-2; adapte-o para
dados reais apontando o `VideoClipDataset` para o `clips_dir`/`splits.json`
reais em vez da fixture sintética.

Sugestão de redacção para o artigo (subsecção de Implementação/Custo
Computacional): *"O Estágio B agrupa os clipes de um mini-lote ao longo da
dimensão de lote em cada instante temporal, em vez de processar um clipe de
cada vez, reduzindo o número de passagens para a frente de N·T para T por
mini-lote (N = tamanho do lote, T = comprimento da janela); com lote de
tamanho [X] em [hardware], isto reduziu o tempo de relógio em [Y]× e a
memória de pico em [Z]% para uma época, sem alteração na formulação da perda
nem nas métricas reportadas."*

---

## 5. Dados sintéticos: o que existe e o que não existe

**O que existe (só para testar correcção do código, não é dado de
investigação):** `tests/test_train.py::_build_synthetic_clips` gera imagens
de ruído uniforme aleatório e uma caixa delimitadora fixa e arbitrária por
clipe. O único propósito é provar que os gradientes fluem e que o mecanismo
de lote funciona; **não tem qualquer semelhança com filmagem real de
vigilância** e não deve ser usado como base de nenhuma experiência reportada
nem como semente para aumento de dados.

**O que ainda não existe (dados sintéticos de nível de investigação):** o
gerador de injecção de ataques ao canal discutido para a linha de
convergência físico-cibernética SENTRY-C (`sentryc/simulate.py`, ainda no
plano, não implementado): extrair clipes reais do UCF-Crime/Le2i e injectar
anomalias declaradas e parametrizadas (congelamento, replay, splice,
perda de frames, blackout, adulteração de timestamp) com um parâmetro de
acoplamento declarado e negativos difíceis explícitos (perda de pacote real,
artefactos de compressão, cenas nocturnas genuinamente estáticas). Este é o
próximo artefacto de código a construir para essa linha de investigação, não
algo já entregue.

---

## 6. Actualização do notebook (célula [5] reescrita)

A célula [5] do `notebooks/sentry_kaggle.ipynb` (Estágio B) foi reescrita
para importar `build_criterion`/`run_stage_b` de `sentry.train` e
`collate_batched` de `sentry.data`, em vez de reimplementar o loop de treino
inline. Isto significa que a célula do notebook agora beneficia de:

- **Tolerância a versões do Ultralytics e a gerações do YOLO** (secção 2) —
  se `BEST_A` vier de um YOLOv8, YOLO11 ou YOLO26, a célula funciona sem
  alteração.
- **Processamento em lote** — ganho de tempo/memória esperado no T4 do
  Kaggle, na mesma linha do medido em CPU (secção 4).
- **Falha ruidosa** em vez de sucesso silencioso, caso uma época não tenha
  nenhum frame rotulado.

A célula [6] (avaliação do Nível S) **não precisou de alterações**: continua
a carregar `TFM_CKPT` da mesma forma, e o nome dessa variável foi mantido
exactamente igual, por compatibilidade.

**Antes de confiar em novos números desta célula**, corra-a no Kaggle e
confirme que o padrão da Tabela 2 se mantém (a perda de treino a cair
monotonicamente, o F1 de teste a degradar) — deve manter-se, porque a
matemática não mudou, só a forma como as chamadas ao modelo são feitas.

---

## 7. Itens em aberto / próximas decisões

- [x] ~~Decidir se a célula [5] do notebook passa a usar o loop em lote~~ —
      feito.
- [x] ~~Verificar compatibilidade com YOLO11/YOLO26~~ — feito, verificado ao
      vivo para os três.
- [ ] Correr a comparação de lote à escala do artigo no Kaggle, para obter
      números citáveis (protocolo da secção 4).
- [ ] Decidir sobre o segundo anotador e começar a calibração da Fase 0 do
      benchmark de vídeo PSID-8 (não relacionado com esta auditoria, continua
      a ser o bloqueio para o Artigo 2).
- [ ] `sentryc/simulate.py` (gerador sintético de ataques ao canal) continua
      por construir; construir apenas depois de o trabalho do benchmark
      PSID-8 estar em curso, conforme a ordem de artigos acordada
      (Artigo 2 → 3 → 4).


---

## 8. Ronda no Kaggle: ablação de arquitectura YOLOv8m vs. YOLO26m (Opção C, decidida)

Decisão tomada: o Artigo 1 mantém-se como está, com os números do `yolov8m`
já reportados (Tabela 2, Secção 5.1, 5.4). O `yolo26m` entra como uma
**ablação de arquitectura dentro do mesmo artigo**, testando se o padrão
central (a perda de treino cai monotonicamente enquanto o F1 de teste
degrada) se replica com um detector de cabeça dupla, sem DFL. Se replicar,
a conclusão "o gargalo é dado, não arquitectura" deixa de depender de uma
única arquitectura e fica muito mais robusta. Também produz o número de
latência de borda que sustenta a direcção de convergência físico-cibernética
da tese (cabeça `one2one` sem NMS).

**Descoberta adicional durante a preparação desta ronda:** a saída do modelo
em modo avaliação (não treino) tem um problema semelhante ao já resolvido
para o modo treino, mas mais traiçoeiro. Nesta versão do Ultralytics, **tanto
o YOLOv8 como o YOLO26 devolvem a mesma forma** em modo avaliação (`(tensor,
dict)`), pelo que distinguir por tipo ou forma do tensor **não é seguro** —
uma primeira tentativa de fazê-lo produziu pontuações fora de `[0,1]`
silenciosamente (sem erro) para o YOLOv8, um bug que só um teste dedicado
apanhou. A correcção definitiva delega no próprio `non_max_suppression` do
Ultralytics com o parâmetro `end2end=`, a mesma via oficial que
`DetectionPredictor.postprocess` usa — exactamente o mesmo princípio já
aplicado ao `build_criterion` do treino (secção 2): confiar na bandeira
`end2end` do modelo, nunca inferir pela forma dos tensores. Implementado em
`sentry/ultralytics_adapter.py::decode_eval_output`, testado com os dois
modelos reais (`tests/test_ultralytics_adapter.py`).

### Plano da ronda

**Fase 1 — Estágio A com `yolo26m`, 3 seeds, Le2i/queda.** Use `yolo26m.pt`
(21,9M parâmetros — o par de capacidade do `yolov8m`, 25,9M; **não** use
`yolo26n`, que é 10× menor e não seria uma comparação justa). Mesmo protocolo
de sempre (`run_over_seeds`, seeds `[0,1,2]` pré-registadas).

**Fase 2 — Estágio B com `yolo26m`**, replicando a ablação de épocas
(5/10/30), usando o código já testado com `E2ELoss`. Pergunta central: o
padrão da Tabela 2 replica-se?

**Fase 3 — medir tempo/memória do lote em GPU real** (fecha o item pendente
da secção 4) e a latência do ramo `one2one` sem NMS — o número que sustenta
o argumento de borda.

**Fase 4 — Nível E, em paralelo se quiser expandir classes/vídeos** (UCF-
Crime, zero-shot, não depende da escolha YOLOv8/YOLO26 do SENTRY nem do
Estágio A/B).

**Como não colidir com os resultados existentes:** o notebook foi actualizado
com uma tag `MODEL_TAG` (derivada de `CFG["model_base"]`) que entra em todos
os caminhos de saída (`runs/stageA_fall_{MODEL_TAG}_s{seed}`,
`tfm_{MODEL_TAG}_s0_e{epochs}.pt`, `tierS_fall_{MODEL_TAG}_{tag}.json`), para
que os resultados do `yolov8m` (já publicados) e do `yolo26m` (a nova
ablação) nunca se sobreponham nem se apaguem mutuamente.
