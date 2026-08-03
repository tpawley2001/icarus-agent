#!/usr/bin/env bash
# Icarus installer.
#
# Detects the local inference stack you already run, writes a config pointing
# at it, and puts `icarus` on your PATH. No venv, no build step, no root.
#
#   ./install.sh                 interactive
#   ./install.sh --yes           accept the detected stack, no prompts
#   ./install.sh --detect-only   show what was found and exit
#   ./install.sh --prefix ~/bin  where to link the launcher

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX=""
ASSUME_YES=0
DETECT_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes)      ASSUME_YES=1; shift ;;
    --detect-only) DETECT_ONLY=1; shift ;;
    --prefix)      PREFIX="${2:-}"; shift 2 ;;
    -h|--help)     sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else
  B=""; D=""; G=""; Y=""; R=""; N=""
fi

say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '  %s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

say ""
say "${B}Icarus${N} — a fully local command-line agent"
say "${D}$ROOT${N}"
say ""

# ---------------------------------------------------------------- requirements
say "${B}Requirements${N}"

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
      PY="$cand"; break
    fi
  fi
done
[ -n "$PY" ] || die "Python 3.9+ is required but was not found."
ok "$("$PY" -V 2>&1)"

if ! "$PY" -c 'import yaml' 2>/dev/null; then
  warn "PyYAML is missing — Icarus needs it (it is the only dependency)."
  if [ "$ASSUME_YES" = 1 ]; then REPLY_YN=y; else
    printf '    Install it now with pip --user? [Y/n]: '; read -r REPLY_YN </dev/tty || REPLY_YN=n
  fi
  case "${REPLY_YN:-y}" in
    [Nn]*) die "Install PyYAML and re-run: $PY -m pip install --user pyyaml" ;;
    *) "$PY" -m pip install --user --quiet pyyaml \
         || die "pip failed. Try your package manager, e.g. apt install python3-yaml" ;;
  esac
  "$PY" -c 'import yaml' 2>/dev/null || die "PyYAML still not importable."
fi
ok "PyYAML"
say ""

# ------------------------------------------------------------------- detection
say "${B}Detecting your AI stack${N}"
say ""
DETECT_OUT="$("$PY" -c "
import sys; sys.path.insert(0, '$ROOT')
from icarus import detect
stack = detect.scan()
print(detect.render(stack))
print('---ICARUS-SPLIT---')
import json
print(json.dumps(detect.config_from(stack)))
print(stack.best.base_url if stack.best else '')
" 2>&1)" || die "detection failed:
$DETECT_OUT"

printf '%s\n' "${DETECT_OUT%%---ICARUS-SPLIT---*}"
TAIL="${DETECT_OUT##*---ICARUS-SPLIT---}"
CONFIG_JSON="$(printf '%s' "$TAIL" | sed -n '2p')"
BASE_URL="$(printf '%s' "$TAIL" | sed -n '3p')"

if [ "$DETECT_ONLY" = 1 ]; then exit 0; fi

if [ -z "$BASE_URL" ]; then
  say ""
  warn "No local inference server was found."
  say "    Start one of these and re-run, or point Icarus at it by hand:"
  say "      ${D}llama-swap (recommended — enables /ctx and VRAM planning)${N}"
  say "      ${D}Ollama · LM Studio · vLLM · llama.cpp server · KoboldCpp${N}"
  say ""
  if [ "$ASSUME_YES" = 1 ]; then
    die "nothing to configure"
  fi
  printf '    Enter a base URL to use anyway (blank to abort): '
  read -r MANUAL </dev/tty || MANUAL=""
  [ -n "$MANUAL" ] || die "aborted"
  CONFIG_JSON="$("$PY" -c "
import json,sys
print(json.dumps({'model': {'base_url': sys.argv[1], 'default': ''},
                  'usage': {'enabled': False}}))" "$MANUAL")"
  BASE_URL="$MANUAL"
fi

say ""
if [ "$ASSUME_YES" != 1 ]; then
  printf '  Use %s%s%s? [Y/n]: ' "$B" "$BASE_URL" "$N"
  read -r REPLY_USE </dev/tty || REPLY_USE=y
  case "${REPLY_USE:-y}" in [Nn]*) die "aborted — edit ~/.icarus/config.yaml yourself" ;; esac
fi

# ----------------------------------------------------------------------- write
say ""
say "${B}Configuring${N}"
CFG_PATH="$("$PY" -c "
import sys, json; sys.path.insert(0, '$ROOT')
from icarus import config
import yaml
config.ensure_dirs()
new = json.loads(sys.argv[1])
existing = {}
if config.CONFIG_PATH.exists():
    try:
        existing = yaml.safe_load(config.CONFIG_PATH.read_text()) or {}
    except Exception:
        existing = {}
        print('  ! existing config.yaml was unreadable and has been replaced')
merged = config._merge(existing, new)
header = ('# Icarus — written by install.sh from an autodetected stack.\n'
          '# Only keys set here override the built-in defaults;\n'
          '# run \`icarus --config\` to print the effective configuration.\n\n')
config.CONFIG_PATH.write_text(header + yaml.safe_dump(merged, sort_keys=False))
print(config.CONFIG_PATH)
" "$CONFIG_JSON")" || die "could not write config"
ok "config: $(printf '%s' "$CFG_PATH" | tail -1)"

# ------------------------------------------------------------------------ link
choose_prefix() {
  [ -n "$PREFIX" ] && { printf '%s' "$PREFIX"; return; }
  for d in "$HOME/.local/bin" "$HOME/bin"; do
    case ":$PATH:" in *":$d:"*) printf '%s' "$d"; return ;; esac
  done
  printf '%s' "$HOME/.local/bin"
}
BINDIR="$(choose_prefix)"
mkdir -p "$BINDIR" || die "cannot create $BINDIR"
chmod +x "$ROOT/bin/icarus"
ln -sf "$ROOT/bin/icarus" "$BINDIR/icarus" || die "cannot link into $BINDIR"
ok "launcher: $BINDIR/icarus"

case ":$PATH:" in
  *":$BINDIR:"*) ;;
  *) warn "$BINDIR is not on your PATH — add this to your shell profile:"
     say  "      export PATH=\"\$PATH:$BINDIR\"" ;;
esac

# ---------------------------------------------------------------------- verify
say ""
say "${B}Verifying${N}"
if VER="$("$BINDIR/icarus" --version 2>&1)"; then ok "$VER"; else die "launcher failed: $VER"; fi
if MODELS="$("$BINDIR/icarus" --models 2>&1)"; then
  COUNT="$(printf '%s\n' "$MODELS" | grep -c . || true)"
  ok "$COUNT models reachable"
  printf '%s\n' "$MODELS" | head -4 | sed 's/^/      /'
  [ "$COUNT" -gt 4 ] && say "      ${D}… $((COUNT - 4)) more${N}"
else
  warn "could not list models yet: $(printf '%s' "$MODELS" | head -1)"
fi

say ""
say "${B}Done.${N}  Start with:  ${B}icarus${N}"
say "  ${D}icarus \"summarize this repo\"     one-shot${N}"
say "  ${D}/help    /model    /ctx    /profile    /skills${N}"
say ""
