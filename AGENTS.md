# Agent Instructions

## Principles

1. Fix the immediate task.
2. Record reusable learnings in this file when warranted.
3. Apply prior learnings to the current task.

Update this file when: a mistake is made, the user corrects you, or a clearly better approach is discovered.

### Learning Format

```markdown
## [Category]: [Title]
- **Situation**: When this applies
- **Lesson**: What to do
- **Example**: Concrete example, if useful
```

Categories: `Code Style`, `Patterns`, `Pitfalls`, `Workflow`

---

## Engineering Standards

### Design
- Keep types and functions focused on one responsibility.
- Prefer small modules by domain for new feature areas.
- Depend on typed results and traits over direct printing or tight coupling.
- Compose small checks/workflows instead of monolithic commands.

### Async & I/O
- Add timeouts to external I/O and long-running async operations.
- Run independent async work concurrently.
- Keep side effects at the edges.

### Errors & State
- Return `Result` with useful context.
- Model success, partial success, timeout, and failure states explicitly.
- Do not conflate different failure modes.

### DRY & CLI UX
- Reuse shared output/styling helpers.
- Reuse shared state/result types.
- Keep command output consistent and predictable.

---

## Rust Workflow

### Before Editing
1. Inspect the existing module structure.
2. Find and follow nearby patterns.
3. Identify related modules, tests, and user-facing behavior.

### During Refactoring
- Keep functions small and focused.
- Prefer typed diagnostic results over inline printing.
- Fix closely related bugs when in scope and low-risk.

### After Changes
1. Run `cargo check`.
2. Run `cargo test`.
3. Add or update tests when behavior changes.
4. Verify relevant edge cases.

```bash
cargo check
cargo test
cargo test --lib
cargo test --test integration
cargo test --test e2e
```

---

## Architecture

### Design Goals
- One obvious entrypoint
- Small focused commands
- Fast default startup
- Optional tuning only when asked

### Source Structure

```
src/
├── bin/llmr.rs           # CLI entrypoint
├── lib.rs                 # Library root
├── errors.rs              # Error types
├── cli/                   # CLI (args, commands)
├── docker/                # Docker client
├── models/                # Profile management & GGUF scanning
├── hardware/              # Hardware detection (CPU, GPU, RAM)
├── diagnostics/           # Environment diagnostics
└── utils/                 # Logging, platform, output
```

### Hardware Detection
- Detects CPU, GPU, RAM, NVLink.
- Platform-specific: Linux (`/proc`, `nvidia-smi`), macOS (`sysctl`), Windows (PowerShell/WMI).
- GPU order: NVIDIA → AMD → Intel → Vulkan.

### Docker Integration
- Direct `docker` CLI invocation.
- Auto-selects image by GPU:

| GPU               | Image          |
|-------------------|----------------|
| NVIDIA CUDA >= 550| `server-cuda13`|
| NVIDIA CUDA < 550 | `server-cuda`  |
| AMD               | `server-rocm`  |
| Intel             | `server-intel` |
| Vulkan            | `server-vulkan`|
| CPU-only          | `server`       |

- Container health verified via `/health` endpoint.

---

## Recorded Learnings

### Patterns

**Domain-Driven Modules**
- **Situation**: Adding a new feature area
- **Lesson**: Create a dedicated module tree instead of growing unrelated files

**Typed Diagnostic Results**
- **Situation**: Implementing checks, probes, or detection logic
- **Lesson**: Return typed state structs instead of printing inline
- **Example**:
  ```rust
  #[derive(Debug, Clone)]
  pub struct DiagnosticResult {
      pub success: bool,
      pub data: Option<Info>,
      pub error: Option<String>,
  }
  ```

**Parallel Async Operations**
- **Situation**: Running multiple independent I/O-bound checks
- **Lesson**: Use `tokio::join!` for concurrency
- **Example**: `let (a, b) = tokio::join!(check_a(), check_b());`

**Timeout External Operations**
- **Situation**: Network calls, subprocesses, or async work that may hang
- **Lesson**: Wrap in a timeout with a reasonable bound
- **Example**: `timeout(Duration::from_secs(5), async_operation).await`

**Explicit State Handling**
- **Situation**: Operations with more than one meaningful outcome
- **Lesson**: Represent states explicitly with enums or clear structs

### Pitfalls

**Dry-Run Must Stay Offline**
- **Situation**: Implementing dry-run flows for Docker-backed commands
- **Lesson**: Do not require Docker availability before rendering a dry-run command; only validate dependencies on real execution

**Unit Conversion in Hardware Detection**
- **Situation**: Translating OS-reported memory values into heuristics
- **Lesson**: Normalize units before storing or comparing
- **Example**: Convert Linux `/proc/meminfo` KiB to GiB, Windows RAM bytes to MiB

**Missing Imports in Conditional Compilation**
- **Situation**: Using `cfg` attributes on code blocks referencing types like `Command`
- **Lesson**: Make imports conditional with `#[cfg(...)]`

**Unnecessary Async Markers**
- **Situation**: Marking functions as `async` without `.await` or spawned work
- **Lesson**: Only use `async` when actually needed

### Workflow

**Fix Related Nearby Bugs**
- **Situation**: Finding a clearly related bug in the same area
- **Lesson**: Fix it when scope is small and behavior is well understood

**Prefer High-Signal Tests**
- **Situation**: Writing tests around simple enums, formatting, or conversions
- **Lesson**: Test representative behavior and edge cases instead of one test per trivial branch

## Pitfalls: Failed Tuning Must Not Produce Profiles
- **Situation**: Tuning benchmarks depend on Docker or another external runner
- **Lesson**: Start required services before tuning, propagate benchmark runner errors, and never turn failed benchmark candidates into zero-metric successful profiles
- **Example**: If Docker is installed but the daemon is stopped, `serve` must attempt Docker startup before tuning and only print "Tuning complete" after successful benchmark results are saved

## Patterns: Backend Boundaries Must Be Explicit
- **Situation**: Adding or referencing inference backends beyond llama.cpp
- **Lesson**: Keep planned backends in typed metadata, but reject serve/tune execution until their Docker args, health checks, and tuning profiles are implemented
- **Example**: vLLM and SGLang can appear as planned `Backend` variants, but `Profile::server_args` must not silently reuse llama.cpp flags for them

## Patterns: Startup Readiness Should Poll Fast First
- **Situation**: Waiting for a local server or container to become ready after startup
- **Lesson**: Start with short health-check intervals and bounded request timeouts, then back off to slower polling; avoid coarse fixed sleeps that add avoidable latency after the service is already ready
- **Example**: For `serve`, poll `/health` immediately and every few hundred milliseconds during the initial startup window instead of waiting two seconds between attempts
