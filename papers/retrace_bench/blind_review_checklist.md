# ReTrace-Bench — Blind-Review Checklist

Run through this before exporting any manuscript text from this workspace.

- [ ] **No author names** anywhere in the manuscript text.
- [ ] **No institution / affiliation names.**
- [ ] **No target conference / venue names** (e.g. AAAI, ACL, ICLR, ICML,
  NeurIPS) and no "target / fallback / main track" submission strategy.
- [ ] **Anonymized links only.** Replace any repository URL with
  `[anonymized repository]` and any dataset URL with `[anonymized dataset link]`.
- [ ] **No public usernames / handles** (GitHub org or user, Hugging Face
  namespace) in the manuscript or figure captions.
- [ ] **No self-identifying commit messages or internal session links** quoted
  in the paper text.
- [ ] **No acknowledgements / funding** in the review version.
- [ ] **Figures/tables** carry no embedded repo paths or author metadata in
  captions or exported files.
- [ ] **De-anonymization plan:** the real repository and Hugging Face dataset
  links (kept only in `benchmark/README.md` and the HF card, not in the
  manuscript) can be added back to the camera-ready version after review.

## Where real identifiers legitimately remain (not part of the manuscript)

These are public distribution surfaces, not the blinded paper, so they keep the
real links and must NOT be copied verbatim into the manuscript:

- `benchmark/README.md`
- `release/huggingface/ReTrace-Bench/README.md` (and the live dataset card)
- `docs/retrace_bench/submission_readiness.md` (internal readiness doc)
