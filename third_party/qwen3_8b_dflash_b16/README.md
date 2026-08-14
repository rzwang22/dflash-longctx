# Pinned Qwen3-8B DFlash custom code

This directory vendors the Python custom code and configuration from
`z-lab/Qwen3-8B-DFlash-b16` at immutable revision
`9b41424b7109f9c5413454f481b09a82b85333f4`. The model card declares the MIT
license; the license text is retained in `LICENSE`.

`original/` is the provenance reference. Three files are byte-identical to the
pinned model repository. `utils.py` differs only by a final newline added by the
repository patching workflow; both hashes and that normalization are recorded in
`provenance.json`.

`instrumented/dflash.py` is the Phase 1A implementation. Its default
`spec_generate` path delegates to an algorithmic transcription of the pinned
implementation and returns the same tensor type. Diagnostics and profiling are
available only through new opt-in keyword arguments. The accompanying
`phase1_observability.py` is first-party instrumentation, not upstream code.

No model weight, tokenizer, Hugging Face cache, or generated materialization is
stored here. Use `scripts/materialize_dflash.py` to combine one of these code
variants with weights held outside the Git repository.
