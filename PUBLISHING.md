# Publishing and minting a DOI (GitHub + Zenodo)

Workflow to make the code citable with a permanent identifier.

## Before you start: fill in the marked fields

Replace every `[...]` placeholder in the following files:

- `CITATION.cff` — name, ORCID, affiliation, GitHub username, paper DOI
- `.zenodo.json` — same (metadata Zenodo reads at archiving time)
- `codemeta.json` — same
- `README.md` ("How to cite" section) — BibTeX keys, name, DOI
- `LICENSE` — copyright holder name

## Step by step

1. **Publish on GitHub.** Create the public repository `sentry-psid8` and push
   the code. Confirm that `CITATION.cff`, `.zenodo.json`, `codemeta.json`,
   `LICENSE`, and `README.md` are in the root. GitHub will show the
   "Cite this repository" button.

2. **Connect Zenodo to GitHub.** At https://zenodo.org, log in with your GitHub
   account (profile menu -> GitHub). In the repository list, toggle the switch
   next to `sentry-psid8`. This authorizes Zenodo to archive each new release.

3. **Create a release on GitHub.** In the repository, go to Releases ->
   "Create a new release", set a tag (e.g., `v1.0.0`), and publish. Zenodo
   detects the release, archives the code snapshot, and mints the DOI (usually
   within minutes).

4. **Collect the DOIs.** Zenodo mints two identifiers:
   - **Concept DOI** — always points to the latest version. **Cite this one in
     the paper.**
   - **Version DOI** — specific to the archived release (e.g., v1.0.0).

5. **Update the metadata with the DOI.** Replace
   `10.5281/zenodo.[TO_BE_MINTED]` with the concept DOI in `CITATION.cff`,
   `.zenodo.json`, and `README.md`, then commit.

6. **Cite in the paper.** Use the Zenodo concept DOI in the code/data
   availability section. The GitHub link may be mentioned as the development
   repository, but the archival reference is the Zenodo DOI.

## Notes

- The paper DOI exists only after acceptance/publication; keep the placeholder
  until then and update afterwards.
- For the PSID-8 video benchmark (future work), use a separate Zenodo deposit
  with `upload_type: dataset` and a CC BY 4.0 license.
- Version the metadata alongside the code: each release should reflect the
  correct version in `CITATION.cff` and `codemeta.json`.
