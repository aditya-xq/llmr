#!/usr/bin/env bash
set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-aditya-xq/llmr}"
BINARY_NAME="llmr"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"
STATE_DIR="${STATE_DIR:-${HOME}/.config/${BINARY_NAME}}"
VERSION="${VERSION:-latest}"

CHECK_ONLY=false
DRY_RUN=false
NO_COLOR="${NO_COLOR:-}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check-only) CHECK_ONLY=true ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            cat <<EOF
${BINARY_NAME} installer

Usage: ./install.sh [--dry-run] [--check-only]

Options:
  --dry-run      Show the install/update plan without making changes
  --check-only   Check whether an update is available
  --help, -h     Show this help

Environment:
  VERSION        Version to install, for example 1.2.3. Defaults to latest
  INSTALL_DIR    Install location. Defaults to ~/.local/bin
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if [ -z "$NO_COLOR" ] && [ -t 1 ]; then
    AMBER='\033[38;5;214m'
    TERRACOTTA='\033[38;5;166m'
    SAGE='\033[38;5;108m'
    SLATE='\033[38;5;245m'
    CREAM='\033[38;5;229m'
    BURNT='\033[38;5;208m'
    BOLD='\033[1m'
    DIM='\033[2m'
    NC='\033[0m'
else
    AMBER=''; TERRACOTTA=''; SAGE=''; SLATE=''; CREAM=''; BURNT=''
    BOLD=''; DIM=''; NC=''
fi

log() { printf '%b\n' "$1"; }
label() { printf '%b' "$1$2${NC}"; }
info() { log "  $(label "$AMBER" "→") $1"; }
ok() { log "  $(label "$SAGE" "✓") $1"; }
warn() { log "  $(label "${BOLD}${BURNT}" "!") $1"; }
err() { log "  $(label "${BOLD}${TERRACOTTA}" "✗") $1"; }
step() { log ""; log "  $(label "$AMBER" "→") $(label "${BOLD}${CREAM}" "$1")"; }
dry() { log "    $(label "$SLATE" "dry-run") $1"; }
kv() { log "    $(label "$SLATE" "$1") $(label "$CREAM" "$2")"; }

detect_os() {
    case "$(uname -s)" in
        Linux*) echo "linux" ;;
        Darwin*) echo "macos" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) echo "unsupported" ;;
    esac
}

target_triple() {
    local os="$1"
    local arch="$2"
    case "$os" in
        linux) echo "${arch}-unknown-linux-gnu" ;;
        macos) echo "${arch}-apple-darwin" ;;
        windows) echo "x86_64-pc-windows-msvc" ;;
        *) return 1 ;;
    esac
}

asset_name() {
    local os="$1"
    local arch="$2"
    local target
    target="$(target_triple "$os" "$arch")"

    if [ "$os" = "windows" ]; then
        echo "${BINARY_NAME}-${target}.zip"
    else
        echo "${BINARY_NAME}-${target}.tar.gz"
    fi
}

download_url() {
    local version="$1"
    local os="$2"
    local arch="$3"
    if [ "$version" = "latest" ]; then
        echo "https://github.com/${GITHUB_REPO}/releases/latest/download/$(asset_name "$os" "$arch")"
    else
        echo "https://github.com/${GITHUB_REPO}/releases/download/v${version}/$(asset_name "$os" "$arch")"
    fi
}

verify_checksum() {
    local archive="$1"
    local version="$2"
    local os="$3"
    local arch="$4"

    if ! ensure_command sha256sum; then
        warn "sha256sum not found - skipping checksum verification"
        return 0
    fi

    local sums_url
    if [ "$version" = "latest" ]; then
        sums_url="https://github.com/${GITHUB_REPO}/releases/latest/download/checksums.txt"
    else
        sums_url="https://github.com/${GITHUB_REPO}/releases/download/v${version}/checksums.txt"
    fi

    local asset expected actual
    asset="$(asset_name "$os" "$arch")"
    if ! expected="$(curl -fsSL --retry 3 --retry-delay 2 "$sums_url" | awk -v a="$asset" '$NF == a {print $1}')" || [ -z "$expected" ]; then
        err "Could not fetch or parse checksums.txt for ${asset} - refusing to install"
        return 1
    fi

    actual="$(sha256sum "$archive" | cut -d' ' -f1)"
    if [ "$actual" != "$expected" ]; then
        err "Checksum mismatch for ${asset}"
        err "  expected: ${expected}"
        err "  actual:   ${actual}"
        return 1
    fi
    ok "Checksum verified"
}

current_version() {
    local binary="$1"
    if [ -x "$binary" ]; then
        "$binary" version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true
    fi
}

latest_version() {
    if [ "$VERSION" != "latest" ]; then
        echo "$VERSION"
        return
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "latest"
        return
    fi

    curl -fsSL "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" \
        | grep -oE '"tag_name":[[:space:]]*"v[^"]+"' \
        | head -1 \
        | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'
}

version_gte() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
    [[ "$2" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

ensure_command() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

install_binary() {
    local version="$1"
    local os="$2"
    local arch="$3"
    local url
    local archive
    local display_version
    url="$(download_url "$version" "$os" "$arch")"
    archive="${TMPDIR:-/tmp}/${BINARY_NAME}-${version}-$(asset_name "$os" "$arch")"
    display_version="$version"
    [ "$version" != "latest" ] && display_version="v${version}"

    step "Install ${BINARY_NAME} ${display_version}"
    kv "target      " "$(target_triple "$os" "$arch")"
    kv "install dir " "${INSTALL_DIR}"
    kv "download    " "${url}"

    if [ "$DRY_RUN" = true ]; then
        dry "create ${INSTALL_DIR} and ${STATE_DIR}"
        dry "download release archive"
        dry "extract ${BINARY_NAME} into ${INSTALL_DIR}"
        dry "write ${STATE_DIR}/version"
        return 0
    fi

    mkdir -p "$INSTALL_DIR" "$STATE_DIR"
    curl -fL --retry 3 --retry-delay 2 -o "$archive" "$url"
    verify_checksum "$archive" "$version" "$os" "$arch"

    if [[ "$archive" == *.zip ]]; then
        ensure_command unzip || { err "unzip is required to extract ${archive}"; return 1; }
        unzip -oq "$archive" -d "$INSTALL_DIR"
        [ -f "${INSTALL_DIR}/${BINARY_NAME}.exe" ] && mv -f "${INSTALL_DIR}/${BINARY_NAME}.exe" "${INSTALL_DIR}/${BINARY_NAME}"
    else
        tar -xzf "$archive" -C "$INSTALL_DIR"
    fi

    chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
    rm -f "$archive"
    printf '%s\n' "$version" > "${STATE_DIR}/version"
    ok "Installed ${BINARY_NAME} v${version}"
}

check_docker() {
    if ! ensure_command docker; then
        warn "Docker was not found"
        return 2
    fi
    if docker info >/dev/null 2>&1; then
        ok "Docker is running"
        return 0
    fi
    warn "Docker is installed but not running"
    return 1
}

install_docker_hint() {
    local os="$1"
    if [ "$DRY_RUN" = true ]; then
        dry "show Docker install instructions"
        return 0
    fi

    case "$os" in
        macos)
            info "Docker Desktop: https://www.docker.com/products/docker-desktop/"
            open "https://www.docker.com/products/docker-desktop/" >/dev/null 2>&1 || true
            ;;
        linux)
            if ensure_command apt-get; then
                info "Install Docker with: sudo apt-get update && sudo apt-get install -y docker.io"
            else
                info "Docker install guide: https://docs.docker.com/get-docker/"
            fi
            ;;
    esac
}

check_python() {
    if ensure_command python3; then
        ok "Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) is installed"
        return 0
    fi
    if ensure_command python; then
        ok "Python is installed"
        return 0
    fi
    warn "Python was not found"
    return 2
}

install_python_hint() {
    local os="$1"
    if [ "$DRY_RUN" = true ]; then
        dry "show Python install instructions"
        return 0
    fi

    case "$os" in
        macos)
            if ensure_command brew; then
                info "Install Python with: brew install python"
            else
                info "Python downloads: https://www.python.org/downloads/"
            fi
            ;;
        linux)
            if ensure_command apt-get; then
                info "Install Python with: sudo apt-get update && sudo apt-get install -y python3 python3-pip"
            else
                info "Python downloads: https://www.python.org/downloads/"
            fi
            ;;
    esac
}

add_to_path() {
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*) return 0 ;;
    esac

    local shell_rc=""
    if [ -n "${ZSH_VERSION:-}" ] || [ -f "${HOME}/.zshrc" ]; then
        shell_rc="${HOME}/.zshrc"
    elif [ -f "${HOME}/.bashrc" ]; then
        shell_rc="${HOME}/.bashrc"
    elif [ -f "${HOME}/.profile" ]; then
        shell_rc="${HOME}/.profile"
    fi

    if [ -z "$shell_rc" ]; then
        warn "${INSTALL_DIR} is not on PATH"
        info "Add this to your shell profile: export PATH=\"${INSTALL_DIR}:\$PATH\""
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        dry "append ${INSTALL_DIR} to PATH in ${shell_rc}"
        return 0
    fi

    if ! grep -Fq "${INSTALL_DIR}" "$shell_rc" 2>/dev/null; then
        {
            printf '\n# Added by %s installer\n' "$BINARY_NAME"
            printf 'export PATH="%s:$PATH"\n' "$INSTALL_DIR"
        } >> "$shell_rc"
        info "Added ${INSTALL_DIR} to PATH in ${shell_rc}"
        info "Restart your terminal or run: source ${shell_rc}"
    fi
}

validate_install() {
    local binary="$1"
    if [ "$DRY_RUN" = true ]; then
        dry "run ${binary} version"
        return 0
    fi

    if [ -x "$binary" ]; then
        ok "$("$binary" version 2>/dev/null | head -1) is ready"
        return 0
    fi

    err "Installed binary was not found at ${binary}"
    return 1
}

main() {
    local os arch binary current latest docker_status python_status
    os="$(detect_os)"
    arch="$(detect_arch)"
    binary="${INSTALL_DIR}/${BINARY_NAME}"

    if [ "$os" = "unknown" ] || [ "$arch" = "unsupported" ]; then
        err "Unsupported platform: $(uname -s) / $(uname -m)"
        exit 1
    fi
    if [ "$os" = "windows" ]; then
        err "Use install.ps1 on Windows PowerShell"
        exit 1
    fi

    log ""
    log "  ${BOLD}${AMBER}${BINARY_NAME}${NC} ${CREAM}installer${NC}"
    log "  ${SLATE}repo:${NC} ${GITHUB_REPO}"
    [ "$DRY_RUN" = true ] && warn "Dry run mode: no changes will be made"
    log ""

    current="$(current_version "$binary")"
    current="${current:-0.0.0}"
    latest="$(latest_version)"

    if [ -z "$latest" ]; then
        err "Could not determine latest release version"
        exit 1
    fi

    kv "current     " "${current}"
    kv "latest      " "${latest}"

    if [ "$CHECK_ONLY" = true ]; then
        if [ "$current" != "0.0.0" ] && version_gte "$current" "$latest"; then
            ok "Current version is up to date"
        else
            warn "Update available: ${latest}"
        fi
        exit 0
    fi

    if [ "$current" = "0.0.0" ] || ! version_gte "$current" "$latest"; then
        install_binary "$latest" "$os" "$arch"
    else
        ok "Already on latest version"
    fi

    log ""
    step "Check dependencies"
    docker_status=0
    check_docker || docker_status=$?
    [ "$docker_status" -eq 2 ] && install_docker_hint "$os"

    python_status=0
    check_python || python_status=$?
    [ "$python_status" -eq 2 ] && install_python_hint "$os"

    log ""
    step "Finish"
    validate_install "$binary"
    add_to_path

    log ""
    ok "Done"
    info "Run: $(label "$CREAM" "${BINARY_NAME} serve")"
}

main "$@"
