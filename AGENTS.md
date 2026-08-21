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

## Release & Commit Discipline (REQUIRED)

CI derives releases from git history and `Cargo.toml`. These rules are load-bearing — violating them breaks releases or blocks PRs.

### Commit messages (enforced by the `commit-lint` CI job)
- Format: `<type>(<scope>)?: <imperative summary>` — e.g., `feat(cli): add --port flag`
- Allowed types: `feat` `feature` `fix` `perf` `revert` `chore` `docs` `style` `refactor` `test` `build` `ci`
- Breaking changes: append `!` before the colon (`feat!:`) **or** add a `BREAKING CHANGE: <what>` line in the body. Nothing else produces a major bump.
- Semver impact: `feat` → minor, `fix`/`perf` → patch, everything else → no release impact
- Never use `--no-verify`; never rewrite pushed history on shared branches

### Branches
- Feature work: short-lived branch → PR into `develop`
- Releases: PR from `develop` into `main` — the only permitted source branch (CI-gated)
- Never push directly to `main`

### Versions
- Bump `Cargo.toml` when preparing a release batch, matching what the batch's commits imply: breaking → major, any `feat` → minor, only `fix`/`perf` → patch, chores/docs/ci only → no bump. Commit `Cargo.lock` alongside.
- `release-version` check enforces exact agreement; override only with the `skip-version-check` label on the release PR
- Tags and releases are created by CI from `Cargo.toml`. Do not hand-create tags (`scripts/bump.ps1` on main is the one sanctioned manual path)

### Before every push
```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --locked --lib
```
CI requires all three; failing them wastes a round-trip.

### Workflows you interact with
| Workflow | Trigger | Effect |
|---|---|---|
| `check.yml` | PRs | fmt, clippy `-D warnings`, test compile, unit tests, commit-lint, security audit |
| `gate-main.yml` | PRs to main | rejects source branches other than `develop` |
| `version-check.yml` | PRs to main | validates Cargo.toml against commit-derived semver |
| `tag-release.yml` | push to main | tags unreleased versions, starts release build |
| `sync-develop.yml` | push to main | back-merges main into develop |
| `release.yml` / `release-verify.yml` | new tag / publish | draft release with 5-platform artifacts / URL verification |

Full release procedure: [docs/RELEASING.md](docs/RELEASING.md).

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

## Pitfalls: Release Artifact Names Are a Contract with Installers
- **Situation**: Changing how release assets are named or packaged
- **Lesson**: `install.sh`/`install.ps1` build download URLs from Rust target triples (`llmr-<triple>.tar.gz|zip`, binary at archive root); any producer of release assets must match that contract exactly or users get 404s
- **Example**: `release.ps1` once produced `llmr-windows-x86_64.zip` while installers requested `llmr-x86_64-pc-windows-msvc.zip`

## Workflow: WhatIf Does Not Stop Native Commands
- **Situation**: Adding `-WhatIf` support to PowerShell scripts that call native executables (cargo, git, docker)
- **Lesson**: Cmdlets honor propagated WhatIf preference, but native exes do not; guard native calls behind `if ($WhatIfPreference)` checks

## Workflow: Verify File Existence with git ls-files, Not Glob
- **Situation**: Checking whether repo files (e.g., `.github/` workflows) exist before creating or overwriting them
- **Lesson**: Glob-style tools can skip dot-directories; always confirm with `git ls-files <dir>` before assuming a path is new, or you may silently clobber tracked automation

## Pitfalls: Dependabot Reads Config from the Default Branch
- **Situation**: Changing `dependabot.yml` (e.g., `target-branch`, grouping) on develop only
- **Lesson**: Version-update config is read from the default branch; fixes must land on main before Dependabot behaves differently, or it keeps opening PRs the old way (against main, ungrouped)

## Pitfalls: Workflow Reruns Replay the Original Event Payload
- **Situation**: A check failed, you added a fix (label, env var), then `gh run rerun --failed`
- **Lesson**: Reruns reuse the original event payload, so label/contains checks still see the old state. Query the live API inside the step (`gh pr view --json labels`) or trigger a fresh event (close/reopen, new push)

## Pitfalls: Squash Merges Poison Tag-Based Commit Ranges
- **Situation**: Deriving semver from `git log <last-tag>..HEAD` when develop→main PRs are squash-merged
- **Lesson**: Develop's original commits are never ancestors of main's squashes, so they reappear in every future range. Keep develop == main + new work: after each squash release merge, let Sync Develop land and reset develop onto main if unique history diverges. Release Train labels its own PRs `skip-version-check` because it computes the bump itself

## Patterns: Consolidating Workflows Must Preserve Job Names
- **Situation**: Merging small workflow files into one (e.g., gate-main.yml + version-check.yml → main-gate.yml)
- **Lesson**: Required status checks reference job names, not files; keep `name:` of jobs identical when consolidating to avoid editing branch protection
