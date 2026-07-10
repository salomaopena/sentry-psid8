# Publicação e emissão de DOI (GitHub + Zenodo)

Fluxo para tornar o código citável com um identificador permanente.

## Antes de começar: preencher os campos marcados
Substitua todos os marcadores `[...]` nos seguintes arquivos:
- `CITATION.cff` — nome, ORCID, afiliação, usuário do GitHub, DOI do artigo
- `.zenodo.json` — idem (metadados que o Zenodo lê no arquivamento)
- `codemeta.json` — idem
- `README.md` (seção "Como citar") — chaves BibTeX, nome, DOI
- `LICENSE` — nome do titular do copyright

## Passo a passo

1. **Publicar no GitHub.** Crie o repositório `sentry-psid8` (público) e envie o
   código. Confirme que `CITATION.cff`, `.zenodo.json`, `codemeta.json`, `LICENSE`
   e `README.md` estão na raiz. O GitHub exibirá o botão "Cite this repository".

2. **Conectar o Zenodo ao GitHub.** Em https://zenodo.org, faça login com a conta
   do GitHub (menu do perfil → GitHub). Na lista de repositórios, ative o
   interruptor ao lado de `sentry-psid8`. Isso autoriza o Zenodo a arquivar
   automaticamente cada nova *release*.

3. **Criar uma release no GitHub.** No repositório, vá em Releases → "Create a new
   release", defina uma tag (ex.: `v1.0.0`) e publique. O Zenodo detecta a release,
   arquiva o instantâneo do código e emite o DOI (normalmente em poucos minutos).

4. **Recolher os DOIs.** O Zenodo emite dois identificadores:
   - **Concept DOI** — aponta sempre para a versão mais recente. **É este que se
     cita no artigo.**
   - **Version DOI** — específico da release arquivada (ex.: v1.0.0).

5. **Atualizar os metadados com o DOI.** Substitua `10.5281/zenodo.[A_SER_EMITIDO]`
   pelo concept DOI em `CITATION.cff`, `.zenodo.json` e `README.md`, e faça um novo
   commit. (Opcionalmente, crie uma nova release para arquivar a versão já com o
   DOI embutido.)

6. **Citar no artigo.** Use o concept DOI do Zenodo na seção de disponibilidade de
   código/dados do artigo. O link do GitHub pode ser mencionado como repositório de
   desenvolvimento, mas a referência arquivável é o DOI do Zenodo.

## Observações
- O DOI do artigo (passo dos metadados) só existe após a aceitação/publicação; até
  lá, mantenha o marcador e atualize depois.
- Para o benchmark de vídeo PSID-8 (trabalho futuro), use um depósito Zenodo
  separado com `upload_type: dataset` e licença CC BY 4.0.
- Versione os metadados junto com o código: cada release deve refletir a versão
  correta em `CITATION.cff` e `codemeta.json`.
