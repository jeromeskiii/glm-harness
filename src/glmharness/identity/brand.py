"""Identity brand: the harness's product identity, in one place.

Every runtime identity string (CLI program name, env prefix, log namespace,
user-facing banners/messages) should derive from these constants so the
project can be reconfigured from this single file.
"""

from __future__ import annotations

HARNESS_NAME = "glm-harness"
HARNESS_TITLE = "GLM-5.3-Flash harness"
HARNESS_VERSION = "0.4.5"
LOG_NAMESPACE = "glmharness"
ENV_PREFIX = "GLMH_"
CLI_ENTRY = "glm-harness"

# Ownership: who maintains this repository and where it lives.
HARNESS_AUTHOR = "ohmskiii"
HARNESS_REPO = "https://github.com/jeromeskiii/glm-harness"
HARNESS_HF = "https://huggingface.co/ohmskiii/GLM-5.3-Flash"

# Factual attribution of the model family this harness serves (kept distinct
# from HARNESS_AUTHOR, which is the harness maintainer).
MODEL_VENDOR = "zhipu"
