#!/usr/bin/env bash
# Start the local fact-checking backends (vLLM + Gemma 4 31B FP8, SearXNG).
# Run from the repository root:  bash scripts/start_factcheck_local.sh
#
# - pins the model revision in deploy/factcheck/model.lock (resolved once, then reused;
#   commit that file so every machine serves the same weights)
# - writes deploy/factcheck/.env (model revision + a generated SearXNG secret; not committed)
# - pulls the digest-pinned images, starts them, and records the resolved image IDs
#   in outputs/factcheck/container-images.json (keep it with experiment results)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$ROOT/deploy/factcheck"
COMPOSE=(docker compose -f "$DEPLOY/compose.yaml")
MODEL_REPO="RedHatAI/gemma-4-31B-it-FP8-dynamic"
LOCK="$DEPLOY/model.lock"
ENV_FILE="$DEPLOY/.env"
OUT="$ROOT/outputs/factcheck"
PY="${PYTHON:-python3}"

die() { echo "error: $*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed"
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required (docker compose ...)"
if command -v nvidia-smi >/dev/null; then
  GPUS=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
  echo "GPUs:"; echo "$GPUS" | sed 's/^/  /'
  [ "$(echo "$GPUS" | grep -c .)" -ge 2 ] || die "two GPUs are required for --tensor-parallel-size 2"
else
  die "nvidia-smi not found; install the NVIDIA driver (and NVIDIA Container Toolkit for Docker)"
fi

# 1. Model revision: resolve once, then always reuse the lock file.
if [ -f "$LOCK" ]; then
  REVISION="$(tr -d '[:space:]' < "$LOCK")"
  echo "model revision (from model.lock): $REVISION"
else
  echo "resolving the current revision of $MODEL_REPO ..."
  REVISION="$("$PY" - "$MODEL_REPO" <<'PYEOF'
import json, os, sys, urllib.request
req = urllib.request.Request(f"https://huggingface.co/api/models/{sys.argv[1]}/revision/main")
if os.environ.get("HF_TOKEN"):
    req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
print(json.load(urllib.request.urlopen(req, timeout=30))["sha"])
PYEOF
)" || die "could not resolve the model revision (network or Hugging Face access)"
  [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || die "unexpected revision: $REVISION"
  echo "$REVISION" > "$LOCK"
  echo "pinned model revision $REVISION in deploy/factcheck/model.lock - commit this file"
fi

# 2. .env for Compose (keeps an existing SearXNG secret).
SECRET=""
[ -f "$ENV_FILE" ] && SECRET="$(sed -n 's/^SEARXNG_SECRET=//p' "$ENV_FILE")"
[ -n "$SECRET" ] || SECRET="$("$PY" -c 'import secrets; print(secrets.token_hex(32))')"
umask 077
cat > "$ENV_FILE" <<ENVEOF
FACTCHECK_MODEL_REVISION=$REVISION
SEARXNG_SECRET=$SECRET
ENVEOF
umask 022

# 3. Pull and start.
"${COMPOSE[@]}" pull
"${COMPOSE[@]}" up -d

# 4. Record what actually runs.
mkdir -p "$OUT"
"$PY" - "$OUT/container-images.json" "$REVISION" "$MODEL_REPO" "${COMPOSE[@]}" <<'PYEOF'
import json, subprocess, sys, datetime
out, revision, repo, compose = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
record = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
          "model": {"repo": repo, "revision": revision}, "services": {}}
for svc in ("vlm", "searxng"):
    cid = subprocess.run(compose + ["ps", "-q", svc], capture_output=True, text=True).stdout.strip()
    if not cid:
        record["services"][svc] = {"error": "container not running"}
        continue
    info = json.loads(subprocess.run(["docker", "inspect", cid], capture_output=True, text=True, check=True).stdout)[0]
    image = json.loads(subprocess.run(["docker", "image", "inspect", info["Image"]],
                                      capture_output=True, text=True, check=True).stdout)[0]
    record["services"][svc] = {"configured_image": info["Config"]["Image"], "image_id": info["Image"],
                               "repo_digests": image.get("RepoDigests", []), "container_id": cid,
                               "command": info["Config"].get("Cmd")}
with open(out, "w") as f:
    json.dump(record, f, indent=2)
print(f"recorded resolved images in {out}")
PYEOF

cat <<MSG

Started. The first start downloads the model weights (tens of GB); follow it with:
  docker compose -f deploy/factcheck/compose.yaml logs -f vlm
When the log shows the server is listening, run:
  source deploy/factcheck/local.env.example
  python -m src.factcheck doctor
If loading fails with insufficient KV-cache memory, set --max-model-len 16384 in
deploy/factcheck/compose.yaml, lower the pipeline's --evidence-budget-chars, and record the change.
MSG
