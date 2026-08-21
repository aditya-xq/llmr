# Releasing llmr

This guide explains how releases work — in plain language — for maintainers who cut releases and users who install them.

---

## For Maintainers: How to Cut a Release

### The normal way: one command, then one click

1. **Land your work on develop** — commit in conventional style (`feat:`, `fix:`, ...). That's all the preparation there is; no manual version bumping.
2. **Run the Release Train** — Actions tab → **Release Train** → *Run workflow* (or `gh workflow run release-train.yml`). It:
   - derives the semver bump from conventional commits since the last tag,
   - updates `Cargo.toml` + `Cargo.lock` on develop and pushes `chore(release): vX.Y.Z`,
   - opens (or refreshes) the release PR to `main` with auto-merge armed.
   Once checks pass, the PR merges itself.
3. **Merging triggers everything automatically:**
   - The **Tag Release** workflow sees the new version in `Cargo.toml`, creates the `vX.Y.Z` tag on main, and starts the pipeline.
   - The **Release** workflow builds 5 targets in parallel (Windows x64, Linux x64/arm64, macOS x64/arm64), packages each as `llmr-<target-triple>.<zip|tar.gz>` with sha256 checksums, and opens a **draft** GitHub release.
   - The **Sync Develop** workflow merges main back into develop, so the next cycle starts clean.
4. **You review** — Open the draft on the [Releases page](https://github.com/aditya-xq/llmr/releases), read the notes, and click **Publish** when happy. Nothing is public until you do this.
5. **Automatic verification** — Publishing triggers the **Release Verify** workflow, which checks that every download link actually works.

Release PRs carry the `skip-version-check` label by design: the Release Train itself computed the version from the same commits, so the redundant check is skipped. For hand-made release PRs, leave the label off and CI will verify your `Cargo.toml` bump instead.

Prefer doing it by hand? Bump `Cargo.toml` yourself, open develop→main without the label, and let the **release-version** check validate it. If a release PR was merged without a bump, run `./scripts/bump.ps1 X.Y.Z -Push` on main to cut it manually.

### Choosing major, minor, or patch

You don't guess — CI checks it. The **release-version** required check on every release PR reads all commits since the last tag and computes what the version must be:

| Commits in this batch | Required bump | Example |
|---|---|---|
| Any `BREAKING CHANGE:` footer or `type!:` subject | **major** | `1.4.2` → `2.0.0` |
| At least one `feat:` | **minor** | `1.4.2` → `1.5.0` |
| Only `fix:` / `perf:` (plus chores/docs) | **patch** | `1.4.2` → `1.4.3` |
| Only `chore:` / `docs:` / `ci:` etc. | **no release** — keep version unchanged | `1.4.2` → `1.4.2` |

The PR cannot merge until `Cargo.toml` matches what the commits say — so write commit messages in conventional style (`feat:`, `fix(scope):`, ...) and the right version is always obvious. To override deliberately (e.g., jump straight to a chosen version), add the **`skip-version-check`** label to the release PR.

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

### Optional production hardening (one-time setup)

Everything below is **off by default** — the pipeline stays fully green without it, and each piece activates itself when you add its configuration.

**macOS code signing & notarization** (removes Gatekeeper warnings on macOS):

1. Join the Apple Developer Program; create a **Developer ID Application** certificate
2. Export it as `.p12`, base64-encode it (`base64 -i cert.p12 | pbcopy`)
3. Create an App-Specific Password at appleid.apple.com
4. Add repository secrets: `MACOS_CERTIFICATE`, `MACOS_CERTIFICATE_PASSWORD`, `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID`

The next release build signs with hardened runtime, submits to Apple notarization, and waits for approval. Without the secrets, builds are simply unsigned.

**Homebrew tap** (`brew install aditya-xq/llmr/llmr`):

1. Create a public repo named `homebrew-llmr` under your account (can be empty)
2. Create a fine-grained PAT with read/write `contents` on that repo only
3. Set variable `HOMEBREW_TAP_REPOSITORY` = `aditya-xq/homebrew-llmr` and secret `HOMEBREW_TAP_TOKEN` = the PAT

After every published release, the workflow generates `Formula/llmr.rb` with correct URLs and sha256s and pushes it to the tap.

**Windows Authenticode signing**: deliberately deferred — it requires choosing a certificate vendor (EV cert or Azure Trusted Signing). Decide when Windows SmartScreen warnings become a real user problem; the pipeline will get a conditional step like the macOS one.

---

## For End Users: How to Install and Update

### Install (macOS / Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/aditya-xq/llmr/main/install.sh | sh
```

### Install (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/aditya-xq/llmr/main/install.ps1 | iex
```

Or with cargo-binstall (uses the same release assets):

```bash
cargo binstall llmr
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

The installers verify every download automatically against `checksums.txt` before extracting — a corrupted or tampered archive is refused. To check manually, compare your download's sha256 against `checksums.txt`.

Every release asset also carries a cryptographically signed **build provenance attestation** (proving exactly which commit built it). Verify with:

```bash
gh attestation verify llmr-x86_64-apple-darwin.tar.gz -R aditya-xq/llmr
```

### Build from source instead

```bash
cargo install --path .
```

Requires Rust 1.75+.

---

## Quick Reference

| Task | Command |
|---|---|
| Release a new version | Run **Release Train** workflow (Actions tab or `gh workflow run release-train.yml`) |
| Test the pipeline safely | Run Release workflow manually (dry-run) |
| Check release status | Actions tab → Release / Release Verify workflows |
| Install latest | See installer commands above |
