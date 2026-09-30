#!/usr/bin/env node
/**
 * GLM-5.3-Flash harness — TypeScript entry point (`src/index.ts`).
 *
 * The runtime of this repository lives in `src/glmharness/` (Python; the
 * `glm-harness` console script). This module is the TypeScript front door for
 * Node-based hosts — IDE carriers, DMH controllers, and the sibling "-H"
 * harness family — written in the house runner style: Node >= 22, ESM, zero
 * runtime dependencies, executable directly via type stripping:
 *
 *   node src/index.ts --doctor
 *   node src/index.ts --mock 'hello from ts' 'say hello'
 *   GLMH_TRACE_INIT=1 node src/index.ts --mock ok 'say hello'
 *
 * Initialization sequence (mirrors `src/glmharness/cli.py` step for step):
 *
 *   1. parse argv          — cli.build_parser()                      cli.py:61
 *   2. merge config        — defaults <- GLMH_* env <- CLI flags     cli.py:384
 *   3. validate            — HarnessConfig.validate()                config.py:222
 *   4. resolve runtime     — bridge analog of resolve_model_path():
 *                            $GLMH_HARNESS_BIN -> $GLMH_PYTHON ->
 *                            .venv/bin/glm-harness -> PATH -> python3 -m
 *   5. build child argv    — the _CLI_FIELD_FOR flag mapping         cli.py:387
 *   6. dispatch mode       — doctor | serve | repl | run             cli.py:695
 *
 * Contract kept identical to the Python CLI:
 *
 *   - stdout carries only the final answer (child stdio is inherited);
 *   - every diagnostic the bridge emits goes to stderr;
 *   - exit codes: 0 ok, 2 config error, 3 provider failure,
 *     4 tool/pipeline failure, 130 cancelled.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Execution mode, resolved with the same precedence as `cli.main()`. */
export type HarnessMode = "doctor" | "serve" | "repl" | "oneshot";

/** Value flags the bridge understands; values stay raw strings until validate. */
export interface RawFlags {
  mock?: string;
  apiBase?: string;
  apiKey?: string;
  model?: string;
  modelPath?: string;
  session?: string;
  workspace?: string;
  sandbox?: string;
  toolAllowlist?: string;
  maxRounds?: string;
  maxNewTokens?: string;
  reasoningEffort?: string;
  temperature?: string;
  logLevel?: string;
  logFormat?: string;
}

export type FlagKey = keyof RawFlags;

/** Result of step 1 (argv parsing). */
export interface ParsedArgs {
  prompt?: string;
  doctor: boolean;
  repl: boolean;
  serve: boolean;
  traceInit: boolean;
  help: boolean;
  flags: RawFlags;
}

/** Result of steps 2-3 (merged + validated configuration). */
export interface HarnessConfig {
  mode: HarnessMode;
  prompt?: string;
  traceInit: boolean;
  flags: RawFlags;
}

/** Result of steps 4-5: everything needed to execute the child process. */
export interface HarnessSurface {
  /** Absolute path to the executable the child will run. */
  command: string;
  /** Arguments before the harness flags (`-m glmharness.cli` fallback only). */
  prefixArgs: string[];
  /** Harness flags, in Python CLI order, plus mode flags and the prompt. */
  args: string[];
  /** Extra environment layered over `process.env` (PYTHONPATH fallback). */
  env: Record<string, string>;
  /** Working directory for the child (the operator's cwd, never the repo). */
  cwd: string;
  mode: HarnessMode;
  trace?: (line: string) => void;
  describe(): string;
}

export interface BootInput {
  argv: string[];
  env?: NodeJS.ProcessEnv;
  repoRoot?: string;
  stdinIsTty?: boolean;
  trace?: (line: string) => void;
}

export interface BootResult {
  config: HarnessConfig;
  surface: HarnessSurface;
}

/** Config-plane failure: maps onto the Python `ConfigError` exit code (2). */
export class ConfigError extends Error {}

/* ── flag vocabulary ──────────────────────────────────────────────────── */

/** CLI flag -> parsed-field mapping (subset of `cli.build_parser()`). */
const FLAG_SPECS: Record<string, FlagKey> = {
  "--mock": "mock",
  "--api-base": "apiBase",
  "--api-key": "apiKey",
  "--model": "model",
  "--model-path": "modelPath",
  "--session": "session",
  "--workspace": "workspace",
  "--sandbox": "sandbox",
  "--tool-allowlist": "toolAllowlist",
  "--max-rounds": "maxRounds",
  "--max-new-tokens": "maxNewTokens",
  "--reasoning-effort": "reasoningEffort",
  "--temperature": "temperature",
  "--log-level": "logLevel",
  "--log-format": "logFormat",
};

/** Parsed field -> `GLMH_*` environment variable (layering step 2). */
const ENV_SPECS: Record<FlagKey, string> = {
  mock: "GLMH_MOCK",
  apiBase: "GLMH_API_BASE",
  apiKey: "GLMH_API_KEY",
  model: "GLMH_MODEL",
  modelPath: "GLMH_MODEL_PATH",
  session: "GLMH_SESSION",
  workspace: "GLMH_WORKSPACE",
  sandbox: "GLMH_SANDBOX",
  toolAllowlist: "GLMH_TOOL_ALLOWLIST",
  maxRounds: "GLMH_MAX_ROUNDS",
  maxNewTokens: "GLMH_MAX_NEW_TOKENS",
  reasoningEffort: "GLMH_REASONING_EFFORT",
  temperature: "GLMH_TEMPERATURE",
  logLevel: "GLMH_LOG_LEVEL",
  logFormat: "GLMH_LOG_FORMAT",
};

/** Parsed field -> Python CLI flag emitted to the child (step 5). */
const PY_FLAGS: Record<FlagKey, string> = {
  mock: "--mock",
  apiBase: "--api-base",
  apiKey: "--api-key",
  model: "--model",
  modelPath: "--model-path",
  session: "--session",
  workspace: "--workspace",
  sandbox: "--sandbox",
  toolAllowlist: "--tool-allowlist",
  maxRounds: "--max-rounds",
  maxNewTokens: "--max-new-tokens",
  reasoningEffort: "--reasoning-effort",
  temperature: "--temperature",
  logLevel: "--log-level",
  logFormat: "--log-format",
};

/** Deterministic emission order for the child argv. */
const FLAG_ORDER: FlagKey[] = [
  "mock",
  "apiBase",
  "apiKey",
  "model",
  "modelPath",
  "session",
  "workspace",
  "sandbox",
  "toolAllowlist",
  "maxRounds",
  "maxNewTokens",
  "reasoningEffort",
  "temperature",
  "logLevel",
  "logFormat",
];

const USAGE = `glm-harness (TypeScript entry) — run GLM-5.3-Flash through the H1 plugin harness.

Usage:
  node src/index.ts [flags] [prompt]

Modes (resolved like cli.py: doctor > serve > repl > one-shot):
  (prompt given)                     one-shot turn; the answer is the only stdout line
  (no prompt, tty stdin)             interactive REPL (delegates to --repl)
  --doctor                           pre-flight: config, snapshot, inference extra
  --repl                             interactive multi-turn REPL
  --serve                            stdio JSON-RPC 2.0 protocol server

Flags (layered: built-in defaults <- GLMH_* env <- these flags):
  --mock TEXT                        deterministic response, no model weights needed
  --api-base URL                     OpenAI-compatible endpoint (vLLM/SGLang/Ollama/API)
  --api-key KEY                      bearer token for the endpoint
  --model ID                         remote model identifier (default GLM-5.3-Flash)
  --model-path DIR                   local snapshot directory
  --session PATH                     append-only JSONL session log
  --workspace DIR                    workspace root for filesystem/bash tools
  --sandbox allow|deny|ask           mutating-tool policy (default deny)
  --tool-allowlist a,b,c             restrict callable tools
  --max-rounds N                     tool-calling rounds per turn
  --max-new-tokens N                 per-request token budget
  --reasoning-effort low|high|max    thinking budget
  --temperature F                    sampling temperature in [0.0, 2.0]
  --log-level LEVEL                  INFO, DEBUG, ...
  --log-format text|json             stderr diagnostic format

Bridge flags:
  --trace-init                       trace the initialization sequence to stderr
  -h, --help                         this help

Environment (bridge-specific):
  GLMH_HARNESS_BIN                   explicit path to the harness console script
  GLMH_PYTHON                        python interpreter for the -m glmharness.cli fallback
  GLMH_TRACE_INIT=1                  same as --trace-init

Runtime resolution order: $GLMH_HARNESS_BIN -> $GLMH_PYTHON (-m) ->
<repo>/.venv/bin/glm-harness -> glm-harness on PATH -> python3 -m glmharness.cli.

Exit codes: 0 success | 2 configuration error | 3 provider failure |
4 tool/pipeline failure | 130 cancelled.
`;

/* ── step 1: argv parsing ─────────────────────────────────────────────── */

/** Parse argv into flags + at most one positional prompt (argparse-shaped). */
export function parseArgs(argv: string[]): ParsedArgs {
  const parsed: ParsedArgs = {
    doctor: false,
    repl: false,
    serve: false,
    traceInit: false,
    help: false,
    flags: {},
  };
  const positionals: string[] = [];
  let index = 0;
  while (index < argv.length) {
    const token = argv[index];
    if (token === "--") {
      // argparse convention: everything after ``--`` is positional.
      positionals.push(...argv.slice(index + 1));
      break;
    }
    if (token === "-h" || token === "--help") {
      parsed.help = true;
      index += 1;
      continue;
    }
    if (token === "--doctor" || token === "--repl" || token === "--serve" || token === "--trace-init") {
      if (token === "--doctor") {
        parsed.doctor = true;
      } else if (token === "--repl") {
        parsed.repl = true;
      } else if (token === "--serve") {
        parsed.serve = true;
      } else {
        parsed.traceInit = true;
      }
      index += 1;
      continue;
    }
    if (token.startsWith("-")) {
      const eq = token.indexOf("=");
      const flag = eq === -1 ? token : token.slice(0, eq);
      const key = FLAG_SPECS[flag];
      if (key === undefined) {
        throw new ConfigError(
          `unknown flag: ${flag} (run 'node src/index.ts --help', or 'glm-harness --help' for the full surface)`,
        );
      }
      let value: string;
      if (eq !== -1) {
        value = token.slice(eq + 1);
      } else {
        const next = argv[index + 1];
        if (next === undefined) {
          throw new ConfigError(`flag ${flag} expects a value`);
        }
        value = next;
        index += 1;
      }
      parsed.flags[key] = value;
      index += 1;
      continue;
    }
    positionals.push(token);
    index += 1;
  }
  if (positionals.length > 1) {
    throw new ConfigError(
      `expected at most one prompt argument, got ${positionals.length} (quote the prompt)`,
    );
  }
  if (positionals.length === 1) {
    parsed.prompt = positionals[0];
  }
  return parsed;
}

/* ── steps 2-3: merge config, validate ────────────────────────────────── */

/**
 * Apply the same layering as `cli._merge_config`: built-in defaults (owned by
 * the Python side) <- `GLMH_*` environment <- explicit CLI flags.
 */
export function mergeConfig(parsed: ParsedArgs, env: NodeJS.ProcessEnv): HarnessConfig {
  const flags: RawFlags = { ...parsed.flags };
  for (const key of FLAG_ORDER) {
    if (flags[key] !== undefined) {
      continue; // an explicit flag always wins over the environment
    }
    const raw = env[ENV_SPECS[key]];
    if (typeof raw === "string" && raw.length > 0) {
      flags[key] = raw;
    }
  }
  const traceInit = parsed.traceInit || env.GLMH_TRACE_INIT === "1";
  return {
    mode: resolveMode(parsed, env),
    prompt: parsed.prompt,
    traceInit,
    flags,
  };
}

/** Mode precedence mirrors `cli.main()`: doctor > serve > repl > one-shot. */
function resolveMode(parsed: ParsedArgs, env: NodeJS.ProcessEnv): HarnessMode {
  if (parsed.doctor) {
    return "doctor";
  }
  if (parsed.serve) {
    return "serve";
  }
  if (parsed.repl) {
    return "repl";
  }
  if (parsed.prompt === undefined && process.stdin.isTTY === true) {
    // cli.py:700 — a bare invocation on a tty falls through to the REPL.
    return "repl";
  }
  return "oneshot";
}

/** Fail-fast validation mirroring `HarnessConfig.validate()` (config.py:222). */
export function validate(config: HarnessConfig): void {
  const { flags } = config;
  const choices: Array<[FlagKey, readonly string[]]> = [
    ["sandbox", ["allow", "deny", "ask"]],
    ["reasoningEffort", ["low", "high", "max"]],
    ["logFormat", ["text", "json"]],
  ];
  for (const [key, allowed] of choices) {
    const value = flags[key];
    if (value !== undefined && !allowed.includes(value)) {
      throw new ConfigError(
        `${key} must be one of ${allowed.join(", ")}, got ${JSON.stringify(value)}`,
      );
    }
  }
  const positiveInts: FlagKey[] = ["maxRounds", "maxNewTokens"];
  for (const key of positiveInts) {
    const value = flags[key];
    if (value === undefined) {
      continue;
    }
    const num = Number(value);
    if (!Number.isInteger(num) || num < 1) {
      throw new ConfigError(`${key} must be an integer >= 1, got ${JSON.stringify(value)}`);
    }
  }
  if (flags.temperature !== undefined) {
    const num = Number(flags.temperature);
    if (!Number.isFinite(num) || num < 0 || num > 2) {
      throw new ConfigError(
        `temperature must be within [0.0, 2.0], got ${JSON.stringify(flags.temperature)}`,
      );
    }
  }
  if (config.mode === "oneshot" && config.prompt !== undefined && config.prompt.trim() === "") {
    throw new ConfigError("prompt is empty");
  }
}

/* ── step 4: runtime resolution ───────────────────────────────────────── */

export interface ResolvedRuntime {
  /** Executable to spawn (absolute when resolved from the repo or env). */
  command: string;
  /** Leading args (`-m glmharness.cli` for the interpreter fallback). */
  prefixArgs: string[];
  /** Extra environment for the child (PYTHONPATH for module execution). */
  env: Record<string, string>;
  /** Human-readable resolution source, quoted by the trace. */
  kind: string;
}

/**
 * Locate the actual harness runtime. The bridge never re-implements the
 * kernel: it finds the same console script an operator would run and lets the
 * Python process own config validation, plugin mounting, and the agent loop.
 */
export function resolveRuntime(repoRoot: string, env: NodeJS.ProcessEnv): ResolvedRuntime {
  const explicitBin = env.GLMH_HARNESS_BIN;
  if (typeof explicitBin === "string" && explicitBin.length > 0) {
    return { command: explicitBin, prefixArgs: [], env: {}, kind: "GLMH_HARNESS_BIN" };
  }
  const explicitPython = env.GLMH_PYTHON;
  if (typeof explicitPython === "string" && explicitPython.length > 0) {
    return {
      command: explicitPython,
      prefixArgs: ["-m", "glmharness.cli"],
      env: { PYTHONPATH: join(repoRoot, "src") },
      kind: "GLMH_PYTHON (-m glmharness.cli)",
    };
  }
  const venvScript = join(repoRoot, ".venv", "bin", "glm-harness");
  if (existsSync(venvScript)) {
    return { command: venvScript, prefixArgs: [], env: {}, kind: "repo .venv console script" };
  }
  const pathVar = env.PATH ?? "";
  for (const dir of pathVar.split(delimiter)) {
    if (dir.length === 0) {
      continue;
    }
    const candidate = join(dir, "glm-harness");
    if (existsSync(candidate)) {
      return { command: candidate, prefixArgs: [], env: {}, kind: "PATH console script" };
    }
  }
  return {
    command: "python3",
    prefixArgs: ["-m", "glmharness.cli"],
    env: { PYTHONPATH: join(repoRoot, "src") },
    kind: "python3 -m glmharness.cli (fallback)",
  };
}

/* ── step 5: child argv assembly ──────────────────────────────────────── */

/** Emit harness flags in CLI order, then the mode flag, then the prompt. */
export function buildChildArgs(config: HarnessConfig, prefixArgs: string[]): string[] {
  const args = [...prefixArgs];
  for (const key of FLAG_ORDER) {
    const value = config.flags[key];
    if (value !== undefined && value !== "") {
      args.push(PY_FLAGS[key], value);
    }
  }
  if (config.mode === "doctor") {
    args.push("--doctor");
  } else if (config.mode === "serve") {
    args.push("--serve");
  } else if (config.mode === "repl") {
    args.push("--repl");
  } else if (config.prompt !== undefined) {
    args.push(config.prompt);
  }
  return args;
}

/* ── steps 2-6: boot ──────────────────────────────────────────────────── */

/** Repo root derived from this module's location (`<repo>/src/index.ts`). */
export function repoRootFrom(moduleUrl: string): string {
  return resolve(dirname(fileURLToPath(moduleUrl)), "..");
}

/**
 * Run the full initialization sequence and return the executable surface.
 * Throws {@link ConfigError} for anything that maps to exit code 2.
 */
export function boot(input: BootInput): BootResult {
  const env = input.env ?? process.env;
  const repoRoot = input.repoRoot ?? repoRootFrom(import.meta.url);
  const parsed = parseArgs(input.argv);
  if (parsed.help) {
    throw new ConfigError("help requested; usage is printed by main() before boot");
  }
  const config = mergeConfig(parsed, env);
  const trace =
    input.trace ??
    (config.traceInit
      ? (line: string): void => {
          process.stderr.write(`[glm-harness.ts] ${line}\n`);
        }
      : undefined);
  trace?.(`init 1/6: parse-args (${config.mode}, ${input.argv.length} token(s))`);
  trace?.("init 2/6: merge-config (defaults <- GLMH_* env <- flags)");
  validate(config);
  trace?.("init 3/6: validate (config ok)");
  const runtime = resolveRuntime(repoRoot, env);
  trace?.(`init 4/6: resolve-runtime (${runtime.kind}: ${runtime.command})`);
  const args = buildChildArgs(config, runtime.prefixArgs);
  trace?.(`init 5/6: build-child-argv (${args.length} arg(s))`);
  const surface: HarnessSurface = {
    command: runtime.command,
    prefixArgs: runtime.prefixArgs,
    args,
    env: runtime.env,
    cwd: process.cwd(),
    mode: config.mode,
    trace,
    describe(): string {
      return `${config.mode} via ${runtime.command} [${runtime.kind}]`;
    },
  };
  return { config, surface };
}

/* ── step 6: dispatch ─────────────────────────────────────────────────── */

const SIGNAL_EXIT_CODES: Record<string, number> = {
  SIGINT: 130,
  SIGTERM: 143,
};

/**
 * Spawn the harness with inherited stdio so the Python CLI's stream contract
 * passes through untouched: stdout carries only the final answer, stderr all
 * diagnostics, and the child's exit code is the bridge's exit code.
 */
export function runHarness(surface: HarnessSurface): Promise<number> {
  surface.trace?.(`init 6/6: dispatch (${surface.describe()})`);
  return new Promise<number>((settle) => {
    const child = spawn(surface.command, surface.args, {
      cwd: surface.cwd,
      env: { ...process.env, ...surface.env },
      stdio: "inherit",
    });
    const forwarded: NodeJS.Signals[] = ["SIGINT", "SIGTERM"];
    const handlers = new Map<NodeJS.Signals, () => void>();
    for (const signal of forwarded) {
      const handler = (): void => {
        if (!child.killed) {
          child.kill(signal);
        }
      };
      handlers.set(signal, handler);
      process.on(signal, handler);
    }
    const cleanup = (): void => {
      for (const [signal, handler] of handlers) {
        process.removeListener(signal, handler);
      }
    };
    child.on("error", (error) => {
      cleanup();
      process.stderr.write(
        `[glm-harness.ts] failed to start the harness runtime (${surface.command}): ${error.message}\n`,
      );
      settle(2); // configuration error: runtime could not be resolved/executed
    });
    child.on("close", (code, signal) => {
      cleanup();
      if (code !== null) {
        surface.trace?.(`exit code ${code}`);
        settle(code);
        return;
      }
      const mapped = signal !== null ? (SIGNAL_EXIT_CODES[signal] ?? 4) : 4;
      surface.trace?.(`terminated by ${signal ?? "unknown signal"} (exit ${mapped})`);
      settle(mapped);
    });
  });
}

/* ── main ─────────────────────────────────────────────────────────────── */

/**
 * Initialize and run the harness. Returns the process exit code; never calls
 * `process.exit` so queued stdout/stderr writes can flush.
 */
export async function main(
  argv: string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env,
): Promise<number> {
  if (argv.includes("-h") || argv.includes("--help")) {
    process.stdout.write(USAGE);
    return 0;
  }
  let booted: BootResult;
  try {
    booted = boot({ argv, env });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    process.stderr.write(`config error: ${detail}\n`);
    return 2;
  }
  return runHarness(booted.surface);
}

/** True when this module was executed directly (not imported by a host). */
function isDirectInvocation(): boolean {
  const entry = process.argv[1];
  if (entry === undefined) {
    return false;
  }
  return resolve(entry) === resolve(fileURLToPath(import.meta.url));
}

if (isDirectInvocation()) {
  main()
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error: unknown) => {
      const detail = error instanceof Error ? error.message : String(error);
      process.stderr.write(`[glm-harness.ts] unhandled failure: ${detail}\n`);
      process.exitCode = 4;
    });
}






