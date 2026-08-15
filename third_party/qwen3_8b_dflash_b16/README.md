# Pinned Qwen3-8B DFlash custom code

This directory vendors the Python custom code and configuration from
`z-lab/Qwen3-8B-DFlash-b16` at immutable revision
`9b41424b7109f9c5413454f481b09a82b85333f4`. The model card declares the MIT
license; the license text is retained in `LICENSE`.

`original/` is the provenance reference. Three files are byte-identical to the
pinned model repository. `utils.py` differs only by a final newline added by the
repository patching workflow; both hashes and that normalization are recorded in
`provenance.json`.

`instrumented/dflash.py` is the opt-in observability/Phase 2C implementation. Its default
`spec_generate` path delegates to an algorithmic transcription of the pinned
implementation and returns the same tensor type. Diagnostics and profiling are
available only through new opt-in keyword arguments. The accompanying
`phase1_observability.py`, materialized `phase2c_online.py`, and materialized
`phase2c_gold.py` are first-party
instrumentation, not upstream code. The latter is copied from
`dflash/phase2c/core.py`; the gold helper is copied from
`dflash/phase2c/gold.py`. Online construction, CPU replay, and Phase 2C.1
classification therefore use repository-controlled definitions.

No model weight, tokenizer, Hugging Face cache, or generated materialization is
stored here. Use `scripts/materialize_dflash.py` to combine one of these code
variants with weights held outside the Git repository.
