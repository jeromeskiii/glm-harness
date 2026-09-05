# Risk map

- **Model shard files** are large immutable artifacts delivered via Git LFS;
  never diff or modify them directly.
- **`TransformersGLM`** requires `transformers>=5.0` and `torch>=2.1`; live
  inference is **not** exercised by unit tests (the module is isolated and
  gated behind the `inference` extra).
- **Tool execution** still defaults to `allow` once a tool registers; a
  sandbox/approval plugin must gate external effects. The kernel exposes
  the seam (`tools/pre-execute`) — write a plugin, don't extend the kernel.
- **Session JSONL** uses an explicit corruption policy (`skip`/`rename`/
  `fail`); schema migrations are not provided — adding a new event type is
  a wire-format change and warrants `corrupt_policy="fail"` pre-deploy.
- **`MockLLM`** is the only provider covered by the standard test gate.
  Provider adapters against real APIs MUST add an env-gated integration
  lane (not run by default).
- **`TransformersGLM.stream`** uses two daemon threads and an asyncio
  queue; if a future adapter adds CPU work on the loop, wrap it with
  `asyncio.to_thread`.
