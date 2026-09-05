"""Integration tests for the LLM adapter.

These tests exercise ``TransformersGLM`` end-to-end. They are skipped by
default because they require the ``[inference]`` extra (transformers +
torch) and (in most cases) a real model snapshot. Opt in by installing
the extra and setting ``GLMH_RUN_INTEGRATION=1``.
"""
