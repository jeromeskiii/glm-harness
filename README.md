# glm-harness

A small, auditable H1-style plugin agent harness for a local
GLM-5.3-Flash snapshot. The kernel owns three primitives —
`Context`, `EventBus`, and `PluginLoader` — and everything else
(session log, tool registry, loop, model adapter) is a replaceable
service behind them.

For full documentation see **[`README-HARNESS.md`](./README-HARNESS.md)**.

For the GLM-5.3-Flash model card and citations see
[`MODEL_CARD.md`](./MODEL_CARD.md).

For the Hugging Face mirror that ships the vendored model
artifacts, see
[ohmskiii/GLM-5.3-Flash](https://huggingface.co/ohmskiii/GLM-5.3-Flash).
