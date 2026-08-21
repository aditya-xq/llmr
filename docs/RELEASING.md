# Releasing llmr

This guide explains how releases work — in plain language — for maintainers who cut releases and users who install them.

---

## For Maintainers: How to Cut a Release

### The normal way: merge your release PR

1. **Prepare develop** — Land all features on develop as usual. When a batch is ready for release, set the version in `Cargo.toml` to the next version (e.g., `1.2.0`).
2. **Raise the release PR** — Open a PR from `develop` to `main`. CI validates the whole batch. Only `develop` may be merged into `main`; any other source branch is blocked by a required check.
3. **Merge it** — Merging triggers everything automatically:
   - The **Tag Release** workflow sees the new version in `Cargo.toml`, creates the `v1.2.0` tag on main, and starts the pipeline.
   - The **Release** workflow builds 5 targets in parallel (Windows x64, Linux x64/arm64, macOS x64/arm64), packages each as `llmr-<target-triple>.<zip|tar.gz>` with sha256 checksums, and opens a **draft** GitHub release.
   - The **Sync Develop** workflow merges main back into develop, so the next cycle starts clean.
4. **You review** — Open the draft on the [Releases page](https://github.com/aditya-xq/llmr/releases), read the notes, and click **Publish** when happy. Nothing is public until you do this.
5. **Automatic verification** — Publishing triggers the **Release Verify** workflow, which checks that every download link actually works.

If you forgot to bump `Cargo.toml` before merging, nothing happens (the tag already exists) — just run `./scripts/bump.ps1 X.Y.Z -Push` on main to cut the release manually.

### Rules to remember

- All changes reach main **through develop**. Direct pushes to main will break the sync job loudly — by design.
- `Cargo.toml` is the single source of truth for versions. Bump it as part of release prep; never edit tags by hand.
- Never edit an already-published release's assets. Cut a new version instead.

### Testing pipeline changes (without releasing)

In the GitHub Actions tab, run the **Release** workflow manually with dry-run enabled. It builds and packages everything but never publishes.

### If something fails

- **Build failure on one platform**: check the Actions log for that matrix job; fix and push a new tag.
- **Wrong version tagged**: delete the tag (`git push origin :refs/tags/vX.Y.Z` and `git tag -d vX.Y.Z`), fix the version, re-tag. Draft releases from failed attempts should be deleted manually.
- **A download link is broken after publishing**: Release Verify will show exactly which asset failed.

---

## For End Users: How to Install and Update

### Install (macOS / Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/aditya-xq/llmr/develop/install.sh | sh
```

### Install (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/aditya-xq/llmr/develop/install.ps1 | iex
```

The installer figures out your operating system and processor type automatically, downloads the right prebuilt binary from the latest GitHub release, puts it on your PATH, and checks that Docker and Python are available.

### Update

Just run the same command again — or use `llmr update`. The installer compares your installed version with the latest release and upgrades if needed.

### Pin a specific version

```bash
VERSION=1.2.3 ./install.sh        # macOS / Linux
$env:VERSION = "1.2.3"; .\install.ps1   # Windows PowerShell
```

### Verify a download (optional)

Every release includes a `checksums.txt` file. After downloading an archive, compare its sha256 hash against that file to confirm it wasn't corrupted or tampered with.

### Build from source instead

```bash
cargo install --path .
```

Requires Rust 1.75+.

---

## Quick Reference

| Task | Command |
|---|---|
| Release a new version | `./scripts/bump.ps1 X.Y.Z -Push` |
| Test the pipeline safely | Run Release workflow manually (dry-run) |
| Check release status | Actions tab → Release / Release Verify workflows |
| Install latest | See installer commands above |
