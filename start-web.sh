#!/usr/bin/env bash
# Host launcher: one V100 queue shared by localhost, LAN and Tailscale.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing project virtual environment: $PROJECT_DIR/.venv/bin/python" >&2
  exit 1
fi
# Listen addresses: 127.0.0.1 always, plus this machine's own LAN and
# Tailscale IPv4 addresses when it has them. The server refuses to start if any
# address it is given cannot be bound, so only addresses found on THIS machine
# are passed. QWEN_LAN_HOST / QWEN_TAILSCALE_HOST choose an address explicitly;
# set either to "" to leave that network out.
is_private_ipv4() {
  [[ $1 =~ ^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)[0-9.]+$ ]]
}
is_tailscale_ipv4() {
  [[ $1 =~ ^100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+$ ]]
}
if [[ -z ${QWEN_LAN_HOST+set} ]]; then
  # The source address of the default route, i.e. the LAN interface.
  QWEN_LAN_HOST="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -o 'src [0-9.]*' | cut -d' ' -f2 || true)"
  is_private_ipv4 "$QWEN_LAN_HOST" || QWEN_LAN_HOST=""
fi
if [[ -z ${QWEN_TAILSCALE_HOST+set} ]]; then
  QWEN_TAILSCALE_HOST="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  is_tailscale_ipv4 "$QWEN_TAILSCALE_HOST" || QWEN_TAILSCALE_HOST=""
fi
qwen_args=(
  -m qwen_engine.web --enable-generation
  --host 127.0.0.1
)
[[ -n $QWEN_LAN_HOST ]] && qwen_args+=(--host "$QWEN_LAN_HOST")
[[ -n $QWEN_TAILSCALE_HOST ]] && qwen_args+=(--host "$QWEN_TAILSCALE_HOST")
qwen_args+=(
  --port "${QWEN_WEB_PORT:-8765}"
  --manifest models/manifest.json --models-dir models/assets
  --q8-manifest models/q8_manifest.json
  --output-dir output/web --cache-dir cache
)
# The remote listener (C1): remote invited users only, reached by Railway's
# reverse proxy over Tailscale. Enabled only when a 0600 token file exists (or
# QWEN_REMOTE_TOKEN_FILE names one) AND the public origin is known; all four
# flags go together or not at all. Everything stays on the local port: admin,
# Settings, model downloads and cross-user "delete all history" answer 404
# here. QWEN_REMOTE_HOST overrides the Tailscale address reported by
# `tailscale ip -4`; set QWEN_REMOTE=0 to leave the listener out anyway.
remote_token_file="${QWEN_REMOTE_TOKEN_FILE:-cache/remote-token}"
remote_origin="${QWEN_REMOTE_ORIGIN:-}"
remote_tailscale_ip="${QWEN_REMOTE_HOST:-}"
if [[ ${QWEN_REMOTE:-1} == 1 && -n $remote_origin && -f $remote_token_file ]]; then
  if [[ -z $remote_tailscale_ip ]] && command -v tailscale >/dev/null 2>&1; then
    # `|| true`: a tailscaled that is down must leave the listener off, not
    # abort the launcher under `set -e`.
    remote_tailscale_ip="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -n $remote_tailscale_ip ]]; then
    qwen_args+=(
      --remote-port "${QWEN_REMOTE_PORT:-8766}"
      --remote-host "$remote_tailscale_ip"
      --remote-token-file "$remote_token_file"
      --remote-origin "$remote_origin"
    )
    # QWEN_REMOTE_AUTO_INVITE=1: anyone who gets through Google sign-in gets
    # an account (Q8 only) on first visit; Google's Test users list is then
    # the only allowlist. People removed on the Admin tab stay out.
    if [[ ${QWEN_REMOTE_AUTO_INVITE:-0} == 1 ]]; then
      qwen_args+=(--remote-auto-invite)
    fi
  else
    echo "notice: no Tailscale IPv4 address found; the remote listener is off" >&2
  fi
fi
# This launcher targets the direct host environment used for CLI validation.
# Set to 0 when running under a cgroup RAM cap; normal encoder scope checks apply.
if [[ ${QWEN_ATTEST_UNBOUNDED_HOST:-1} == 1 ]]; then
  qwen_args+=(--attest-unbounded-host)
fi
# Skip re-hashing model files whose size and mtime are unchanged since they
# were last verified (saves ~20 s per job). Set to 0 to hash every job.
if [[ ${QWEN_REUSE_VERIFIED_ASSETS:-1} == 1 ]]; then
  qwen_args+=(--reuse-verified-assets)
fi
# Prompt rewriting. Two rewriters exist and BOTH may be configured: the
# official enhancers below (the standard ones, which briefly take the GPU) and
# the shared local Qwen3.8 service (which never touches the GPU). The upstream
# system prompts are licence-restricted, so they live in the ignored
# models/rewrite/ and are digest-checked at startup. Which rewriter answers a
# TEXT draft and which answers an EDIT draft are user settings, chosen on the
# web page's Settings tab and stored by the server; these flags only say what is
# possible and seed the first start.
rewrite_dir=models/rewrite
rewrite_prompt_t2i_passed=0
rewrite_prompt_edit_passed=0
if [[ ${QWEN_REWRITE:-1} == 1 && -f $rewrite_dir/system_prompt_t2i.txt \
      && -f $rewrite_dir/system_prompt_edit.txt ]]; then
  qwen_args+=(
    --rewrite-backend local-llm
    --rewrite-endpoint "${QWEN_REWRITE_ENDPOINT:-http://127.0.0.1:8079/v1/chat/completions}"
    --rewrite-model "${QWEN_REWRITE_MODEL:-Qwen3.8}"
    --rewrite-prompt-t2i "$rewrite_dir/system_prompt_t2i.txt"
    --rewrite-prompt-edit "$rewrite_dir/system_prompt_edit.txt"
  )
  rewrite_prompt_t2i_passed=1
  rewrite_prompt_edit_passed=1
fi
# The official Qwen-Image-2.1 enhancers, each run as a Q8_0 GGUF under
# llama-server for the duration of one rewrite. PE-T2I answers a text draft;
# PE-I2I answers an edit draft and is useless without its vision projector
# (mmproj), so its two files are passed together or not at all. Each pair is
# offered only when its own files are readable, which means a half-installed
# models/pe/ still gets the enhancer it can actually run — and an edit draft
# falls back to the endpoint, with the Settings page saying why. Every
# enhancer reads the same digest-pinned system prompts as the endpoint, so those
# files are required too. Set QWEN_PE=0 to leave both out even when present.
pe_binary="${QWEN_PE_LLAMA_SERVER:-$HOME/Sandbox/llama.cpp-build/llama.cpp/build/bin/llama-server}"
# Unset and not at the default location: use a llama-server on PATH, if any.
if [[ -z ${QWEN_PE_LLAMA_SERVER:-} && ! -x $pe_binary ]]; then
  pe_binary="$(command -v llama-server || true)"
fi
pe_model="${QWEN_PE_MODEL:-models/pe/PE-T2I-Q8_0/Qwen-Image-2.1-PE-T2I-Q8_0.gguf}"
pe_i2i_model="${QWEN_PE_I2I_MODEL:-models/pe/PE-I2I-Q8_0/Qwen-Image-2.1-PE-I2I-Q8_0.gguf}"
pe_i2i_mmproj="${QWEN_PE_I2I_MMPROJ:-models/pe/PE-I2I-Q8_0/mmproj-Qwen-Image-2.1-PE-I2I-BF16.gguf}"
if [[ ${QWEN_PE:-1} == 1 && -x "$pe_binary" && -r "$pe_model" \
      && -f $rewrite_dir/system_prompt_t2i.txt ]]; then
  qwen_args+=(--pe-llama-server "$pe_binary" --pe-t2i-model "$pe_model")
  if [[ $rewrite_prompt_t2i_passed == 0 ]]; then
    qwen_args+=(--rewrite-prompt-t2i "$rewrite_dir/system_prompt_t2i.txt")
  fi
  if [[ -r "$pe_i2i_model" && -r "$pe_i2i_mmproj" \
        && -f $rewrite_dir/system_prompt_edit.txt ]]; then
    qwen_args+=(--pe-i2i-model "$pe_i2i_model" --pe-i2i-mmproj "$pe_i2i_mmproj")
    if [[ $rewrite_prompt_edit_passed == 0 ]]; then
      qwen_args+=(--rewrite-prompt-edit "$rewrite_dir/system_prompt_edit.txt")
    fi
  fi
  if [[ -n ${QWEN_PE_TIMEOUT:-} ]]; then
    qwen_args+=(--pe-timeout "$QWEN_PE_TIMEOUT")
  fi
fi
# Machine names browsers may type instead of an IP address live in
# cache/web-settings.json ({"host_names": [...]}), read at startup. The first
# run writes it with this PC's hostname, its mDNS name (<hostname>.local) and,
# when Tailscale is up, its MagicDNS name. Edit it and restart to change them;
# an empty list accepts IP addresses and localhost only.
if [[ ! -e cache/web-settings.json ]]; then
  short_name="$(hostname -s 2>/dev/null || true)"
  magic_dns=""
  if command -v tailscale >/dev/null 2>&1; then
    magic_dns="$(tailscale status --self --json 2>/dev/null \
      | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"])' 2>/dev/null || true)"
  fi
  mkdir -p cache
  .venv/bin/python - "$short_name" "${magic_dns%.}" > cache/web-settings.json <<'PY'
import json, sys
short, magic = (arg.lower() for arg in sys.argv[1:])
names = [short, short + ".local", magic] if short else [magic]
print(json.dumps({"host_names": [name for name in names if name]}, indent=2))
PY
fi
exec .venv/bin/python "${qwen_args[@]}" "$@"
