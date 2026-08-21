# Releasing llmr

This guide explains how releases work — in plain language — for maintainers who cut releases and users who install them.

---

## For Maintainers: How to Cut a Release

You need one command:

```powershell
./scripts/bump.ps1 1.2.3 -Push
```

That's it. Here is what happens, step by step:

1. **Version bump** — The script sets the version in `Cargo.toml` to `1.2.3`, refreshes `Cargo.lock`, commits as `chore(release): v1.2.3`, creates an annotated git tag `v1.2.3`, and pushes both to GitHub.
2. **CI validates** — Pushing the tag starts the **Release** workflow. First it checks that the tag matches the version in `Cargo.toml` (if they don't match, everything stops — no bad release can happen) and that the code compiles.
3. **Five builds in parallel** — GitHub's servers build llmr for every supported platform at once:
   - Windows x64
   - Linux x64 and Linux ARM64
   - macOS Intel and macOS Apple Silicon

   Each build produces a package named after its Rust target triple, for example `llmr-x86_64-pc-windows-msvc.zip`, plus a sha256 checksum file.
4. **Draft release** — All packages are collected, a `checksums.txt` is generated, and a **draft** (private, not yet public) GitHub release is created with automatic release notes.
5. **You review** — Open the draft on the [Releases page](https://github.com/aditya-xq/llmr/releases), read the notes, and click **Publish** when happy. Nothing is public until you do this.
6. **Automatic verification** — Publishing triggers the **Release Verify** workflow, which checks that every download link actually works. If any asset is broken, CI turns red immediately.

### Testing pipeline changes (without releasing)

In the GitHub Actions tab, run the **Release** workflow manually with dry-run enabled. It builds and packages everything but never publishes.

### If something fails

- **Build failure on one platform**: check the Actions log for that matrix job; fix and push a new tag.
- **Wrong version tagged**: delete the tag (`git push origin :refs/tags/vX.Y.Z` and `git tag -d vX.Y.Z`), fix the version, re-tag. Draft releases from failed attempts should be deleted manually.
- **A download link is broken after publishing**: Release Verify will show exactly which asset failed.

### Rules to remember

- `Cargo.toml` is the single source of truth for versions. Never edit tags by hand without bumping it.
- Releases only happen through tags. Merging to `main` does nothing by itself.
- Never edit an already-published release's assets. Cut a new version instead.

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
