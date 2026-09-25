#!/bin/bash
# stack029-launch.sh — gated launcher for the upstream-vLLM + B12X image lineage
# (0rand/PILCOTHINK "vllm 0.29 b12x", and our custom builds of the same Dockerfile) on
# the 2× Spark pair, TP=2, via eugr's launch-cluster.sh in --launch-script mode.
# Runs ON the head node (rank 0). Every launch writes a record under ~/stack029/records/.
#
#   start <profile.env>   preflight → render serve.sh → launch → wait /health → record → gates
#   serve <profile.env>   adopt an already-healthy pair, else full start; then monitor
#                         (systemd entry point: stack029-vllm.service runs `serve`)
#   stop                  stop both ranks (launch-cluster.sh stop); boot log saved first
#   gates <profile.env>   re-run the gate battery against the running server
#   status                one screen
#
# Exit: 2 died/wedged in monitor (systemd restarts), 3 preflight refused, 4 launch failed,
# 5 gates failed (server stopped), 64 usage.
set -uo pipefail
HOME_DIR="${STACK029_HOME:-$HOME/stack029}"
EUGR="${EUGR_DIR:-$HOME/spark-vllm-docker-841fdcc}"
RECORDS="$HOME_DIR/records"; GATES="$HOME_DIR/gates"; OVERLAYS="$HOME_DIR/overlays"
CONTAINER="vllm_node"
# Worker (rank 1) reachability. Override for your cluster:
#   WORKER_ADDRS="worker-host worker-ip"   candidates tried in order
#   WORKER_SSH_USER="user"                 account on both nodes (default: current user)
WORKER_ADDRS=(${WORKER_ADDRS:-spark2})
WORKER_SSH_USER="${WORKER_SSH_USER:-$USER}"
SITE="/usr/local/lib/python3.12/dist-packages"
REC=""; LAUNCH_EPOCH=$(date +%s); READY_EPOCH=""; WEDGE_CHECKS="${WEDGE_CHECKS:-5}"   # x 300 s in monitor
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(TS)] $*"; [[ -n "$REC" && -d "$REC" ]] && echo "[$(TS)] $*" >>"$REC/launcher.log" || true; }
die() { local c=$1; shift; log "FATAL: $*"; exit "$c"; }
WORKER=""
pick_worker() { for a in "${WORKER_ADDRS[@]}"; do ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_SSH_USER@$a" true 2>/dev/null && { WORKER="$WORKER_SSH_USER@$a"; return 0; }; done; return 1; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }
health_ok() { [[ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:${PORT:-8000}/health")" == "200" ]]; }
cup() { [[ -n "$(docker ps -q --filter "name=^${CONTAINER}$")" ]]; }
wcup() { [[ -n "$(wssh docker ps -q --filter "name=^${CONTAINER}$" 2>/dev/null)" ]]; }

# ---------- profile defaults = 0rand .env.sample (2026-09-09) ----------
load_profile() {
    IMAGE_REF="stack029/orand:85eb91ee"
    MODEL_ID="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    SERVED_NAMES="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp deepseek-v4-flash"
    PORT=8000; GPU_UTIL=0.87; KV_DTYPE=fp8; MAX_MODEL_LEN=1048576; BLOCK_SIZE=256
    MAX_NUM_SEQS=8; MAX_BATCHED=4096; CAPTURE_MAX=48; CAPTURE_SIZES=""
    K=6; DRAFT_METHOD=probabilistic; ADAPTIVE=false; THINKING=true; EFFORT=max; LIMIT_IMAGES=3
    REVISION="86f746b36186f0e567729a5c06a8c918caba82a9"   # mechanical: our hub cache has no refs/main
    RETENTION=""; FORCE_A16=0; OVERRIDE_GEN=""; PARSER_FIX=0; OPROJ=overlay; EP=0; LPT=""; MMPREFIX_FIX=0
    EXTRA_ARGS=""; EXTRA_ENV=""; RT_MAX_JOBS=16; EXPECT_FP="5,84,97"; READY_TIMEOUT_S=3000
    PROFILE_NAME="$(basename "$1" .env)"
    # shellcheck disable=SC1090
    source "$1"
}

render_serve() { # -> $REC/serve.sh (single vllm serve command; launch-cluster appends --nnodes etc.)
    local rev_args="" spec_rev="" gen="" cap="" ep="" lpt=""
    if [[ -n "$REVISION" ]]; then rev_args="--revision $REVISION --tokenizer-revision $REVISION --code-revision $REVISION"; spec_rev=",\"revision\":\"$REVISION\",\"code_revision\":\"$REVISION\""; fi
    [[ -n "$OVERRIDE_GEN" ]] && gen="--override-generation-config '$OVERRIDE_GEN'"
    if [[ -n "$CAPTURE_SIZES" ]]; then cap="--compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":$CAPTURE_SIZES,\"custom_ops\":[\"all\"]}'"
    else cap="--max-cudagraph-capture-size $CAPTURE_MAX --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}'"; fi
    [[ "$EP" == "1" ]] && ep="--enable-expert-parallel"
    [[ -n "$LPT" ]] && lpt="--long-prefill-token-threshold $LPT"
    cat >"$REC/serve.sh" <<SH
#!/bin/bash
set -euo pipefail
vllm serve ${MODEL_ID} \\
    --served-model-name ${SERVED_NAMES} \\
    --host 0.0.0.0 --port ${PORT} \\
    --trust-remote-code ${rev_args} \\
    --tensor-parallel-size 2 \\
    --kv-cache-dtype ${KV_DTYPE} --block-size ${BLOCK_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN} --max-num-seqs ${MAX_NUM_SEQS} --max-num-batched-tokens ${MAX_BATCHED} \\
    --gpu-memory-utilization ${GPU_UTIL} \\
    --enable-prefix-caching --enable-chunked-prefill --enable-prompt-tokens-details \\
    --skip-mm-profiling --limit-mm-per-prompt '{"image": ${LIMIT_IMAGES}}' \\
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \\
    --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \\
    --default-chat-template-kwargs '{"thinking":${THINKING},"reasoning_effort":"${EFFORT}"}' ${gen} \\
    --moe-backend b12x --linear-backend b12x --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \\
    ${cap} ${ep} ${lpt} ${EXTRA_ARGS} \\
    --speculative-config '{"method":"dspark","model":"${MODEL_ID}","num_speculative_tokens":${K},"draft_sample_method":"${DRAFT_METHOD}","attention_backend":"FLASHINFER_MLA_SPARSE_DSV4","enable_adaptive_verification":${ADAPTIVE}${spec_rev}}'
SH
    chmod +x "$REC/serve.sh"
}

docker_extra_args() { # -> VOL_ENV array
    VOL_ENV=(-e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 -e MAX_JOBS="$RT_MAX_JOBS" -e NVCC_THREADS=4)
    [[ -n "$RETENTION" ]] && VOL_ENV+=(-e "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=$RETENTION")
    [[ "$FORCE_A16" == "1" ]] && VOL_ENV+=(-e VLLM_B12X_MOE_FP4_FORCE_A16=1)
    for kv in $EXTRA_ENV; do VOL_ENV+=(-e "$kv"); done
    [[ "$PARSER_FIX" == "1" ]] && VOL_ENV+=(-v "$OVERLAYS/parser_deepseek_v4__d98c8c0.py:$SITE/vllm/parser/deepseek_v4.py:ro")
    [[ "$OPROJ" == "upstream" ]] && VOL_ENV+=(-v "$OVERLAYS/o_proj__upstream_6fbb00b18.py:$SITE/vllm/models/deepseek_v4/nvidia/ops/o_proj.py:ro")
    # Vision mm-prefix-span fix (V2 model runner): needs the code overlay AND the two
    # config flags injected via --hf-overrides in the profile's EXTRA_ARGS.
    [[ "$MMPREFIX_FIX" == "1" ]] && VOL_ENV+=(
        -v "$OVERLAYS/attn_utils__mmspan.py:$SITE/vllm/v1/worker/gpu/attn_utils.py:ro"
        -v "$OVERLAYS/default__mmspan.py:$SITE/vllm/v1/worker/gpu/model_states/default.py:ro")
    return 0
}

preflight() {
    pick_worker || die 3 "preflight: worker unreachable"
    docker info >/dev/null 2>&1 || die 3 "docker down on head"; wssh docker info >/dev/null 2>&1 || die 3 "docker down on worker"
    local i1 i2; i1=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF" 2>/dev/null) || die 3 "image $IMAGE_REF missing on head"
    i2=$(wssh docker image inspect --format "'{{.Id}}'" "$IMAGE_REF" 2>/dev/null) || die 3 "image $IMAGE_REF missing on worker"
    [[ "$i1" == "$i2" ]] || die 3 "image ID differs head=$i1 worker=$i2"; IMAGE_ID="$i1"
    cup && die 3 "container $CONTAINER already running on head (stop first)"; wcup && die 3 "container $CONTAINER already running on worker"
    local other; other=$(docker ps --format '{{.Names}}' | grep -E 'dspark|vllm' || true); [[ -z "$other" ]] || die 3 "other serving containers on head: $other"
    other=$(wssh docker ps --format "'{{.Names}}'" | grep -E 'dspark|vllm' || true); [[ -z "$other" ]] || die 3 "other serving containers on worker: $other"
    local h w; h=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1); w=$(wssh nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    (( ${h:-0} >= 2950 && ${w:-0} >= 2950 )) || die 3 "clocks.max.sm head=$h worker=$w (<2950 = firmware low-clock; power-cycle)"
    for f in $([[ "$PARSER_FIX" == "1" ]] && echo parser_deepseek_v4__d98c8c0.py) $([[ "$OPROJ" == "upstream" ]] && echo o_proj__upstream_6fbb00b18.py) $([[ "$MMPREFIX_FIX" == "1" ]] && echo attn_utils__mmspan.py default__mmspan.py); do
        [[ -s "$OVERLAYS/$f" ]] || die 3 "overlay missing on head: $f"; wssh test -s "$OVERLAYS/$f" || die 3 "overlay missing on worker: $f"; done
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && log "preflight: page cache dropped (head)" || log "preflight: drop_caches not permitted (head)"
    wssh "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null && log "preflight: page cache dropped (worker)" || log "preflight: drop_caches not permitted (worker)"
    local m1 m2; m1=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); m2=$(wssh "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo")
    (( m1 >= 95 && m2 >= 95 )) || die 3 "MemAvailable head=${m1}G worker=${m2}G (<95G: something still resident)"
    log "preflight: ok worker=$WORKER image=$IMAGE_ID max.sm=$h/$w memavail=${m1}G/${m2}G"
}

launch_stack() {
    docker_extra_args
    log "launch: launch-cluster.sh -t $IMAGE_REF (extra: ${VOL_ENV[*]})"
    # item 5: memlock ulimit -- eugr raises only nofile (issue #389 -> ibv_reg_mr ENOMEM)
    ( cd "$EUGR" && VLLM_SPARK_EXTRA_DOCKER_ARGS="--ulimit memlock=-1:-1 --ulimit stack=67108864" ./launch-cluster.sh -t "$IMAGE_REF" --launch-script "$REC/serve.sh" -d "${VOL_ENV[@]}" ) >"$REC/launch-cluster.log" 2>&1 || { tail -30 "$REC/launch-cluster.log" | sed 's/^/  lc: /'; die 4 "launch-cluster.sh failed"; }
    cp "$EUGR/.env" "$REC/eugr.env"
}
wait_ready() {
    local t0=$SECONDS peak_h=0 peak_w=0 c
    log "wait_ready: /health (timeout ${READY_TIMEOUT_S}s), sampling clocks.sm"
    while (( SECONDS - t0 < READY_TIMEOUT_S )); do
        health_ok && { READY_EPOCH=$(date +%s); log "wait_ready: /health 200 after $((SECONDS-t0))s (boot clock peaks head=$peak_h worker=$peak_w)"; echo "boot_clock_peak_head=$peak_h" >>"$REC/facts.txt"; echo "boot_clock_peak_worker=$peak_w" >>"$REC/facts.txt"; return 0; }
        cup || { docker logs "$CONTAINER" >"$REC/boot.log" 2>&1; die 4 "wait_ready: head container gone during boot (boot.log saved)"; }
        c=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'); (( ${c:-0} > peak_h )) && peak_h=$c
        c=$(wssh 'nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits' 2>/dev/null | head -1 | tr -d ' \r'); (( ${c:-0} > peak_w )) && peak_w=$c
        sleep 15
    done
    docker logs "$CONTAINER" >"$REC/boot.log" 2>&1; die 4 "wait_ready: no /health within ${READY_TIMEOUT_S}s"
}
fill_record() {
    docker logs "$CONTAINER" >"$REC/boot.log" 2>&1 || true
    local kv eng args conc
    kv=$(grep -oE "GPU KV cache size: [0-9,]+" "$REC/boot.log" | head -1 | tr -d ', ' | grep -oE "[0-9]+$" || echo "")
    conc=$(grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$REC/boot.log" | head -1 || echo "")
    eng=$(grep -oE "vLLM API server version [^ ]+|V1 LLM engine \(v[^)]+\)" "$REC/boot.log" | head -1 || echo "")
    args=$(grep -m1 "non-default args" "$REC/boot.log" | cut -c1-4000 || echo "")
    { echo "profile=$PROFILE_NAME"; echo "image_ref=$IMAGE_REF"; echo "image_id=$IMAGE_ID"; echo "kv_pool_tokens=$kv"; echo "max_concurrency=$conc"; echo "engine=$eng"; echo "ready_epoch=$READY_EPOCH"; echo "launch_epoch=$LAUNCH_EPOCH"; } >>"$REC/facts.txt"
    echo "$args" >"$REC/non-default-args.txt"
    log "record: kv_pool_tokens=$kv $conc engine=$eng"
}
nvrm_count() { local since="$1" until="${2:-}"; if [[ -n "$until" ]]; then journalctl -k --since "$since" --until "$until" 2>/dev/null | grep -c NV_ERR_NO_MEMORY; else journalctl -k --since "$since" 2>/dev/null | grep -c NV_ERR_NO_MEMORY; fi; }
gates() {
    local api="http://localhost:${PORT}" fail=0 t0 g_since g_ready
    g_since=$(date -u -d "@$LAUNCH_EPOCH" '+%Y-%m-%d %H:%M:%S UTC'); g_ready=$(date -u -d "@${READY_EPOCH:-$LAUNCH_EPOCH}" '+%Y-%m-%d %H:%M:%S UTC')
    nohup "$GATES/memwatch-swap.sh" "$REC/memwatch-gates-head.tsv" 5 >/dev/null 2>&1 & local mw=$!
    wssh "nohup $GATES/memwatch-swap.sh /tmp/memwatch-gates-worker.tsv 5 >/dev/null 2>&1 & echo \$! > /tmp/memwatch-gates.pid"
    log "gate 1: fingerprints (enforce $EXPECT_FP) + smoke + invalid-level"
    python3 "$GATES/gate-probes.py" --base "$api" --model "$MODEL_ID" --expect-fp "$EXPECT_FP" --fp-mode enforce --out "$REC/gate-probes.json" >"$REC/gate-probes.log" 2>&1 || true
    python3 - "$REC/gate-probes.json" <<'PY' || fail=1
import json, sys
d = json.load(open(sys.argv[1])); fp = d.get("fingerprints", {})
ok = (all(v.get("ok") for v in fp.values())
      and bool(d.get("smoke", {}).get("ok"))
      and bool(d.get("invalid_level", {}).get("ok")))   # item 3: was never inspected
print("fingerprints", "/".join(str(fp[k]["count"]) for k in ("low","high","max") if k in fp), "smoke", d.get("smoke", {}).get("ok"), "invalid_level", d.get("invalid_level"))
sys.exit(0 if ok else 1)
PY
    log "gate 2: basics battery (vision-parametrized)"
    BASICS_MODEL="$MODEL_ID" BASICS_SNAP="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/$REVISION" BASICS_BASE="$api" \
        python3 "$GATES/basics-check-vision.py" >"$REC/basics-check.txt" 2>&1; local be=$?; echo "basics_exit=$be" >>"$REC/facts.txt"; tail -1 "$REC/basics-check.txt" | tee -a "$REC/launcher.log"
        (( be == 0 )) || { log "gate 2: basics battery FAILED (exit $be)"; fail=1; }   # item 3: was recorded but never enforced
    log "gate 3: vision probes"
    python3 "$GATES/vision-probes.py" --base "$api" --model "$MODEL_ID" --imgdir "$GATES" --out "$REC/vision-probes.json" >"$REC/vision-probes.log" 2>&1; local ve=$?; tail -2 "$REC/vision-probes.log" | tee -a "$REC/launcher.log"
    (( ve == 0 )) || { log "gate 3: vision probes FAILED (exit $ve)"; fail=1; }   # item 3: result was never inspected
    log "gate 4: soak c8 mixed 130 s (clocks sampled)"
    ( for i in $(seq 1 12); do sleep 10; echo "$(TS) head=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1) worker=$(wssh 'nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits' 2>/dev/null | head -1)"; done ) >"$REC/soak-clocks.txt" & local clk=$!
    python3 "$GATES/soak-probe.py" --base "$api" --model "$MODEL_ID" --seconds 130 --out "$REC/soak-summary.json" >"$REC/soak.log" 2>&1 || true; wait "$clk"   # F-stack029-1: a bare wait blocked on the memwatch sampler forever (arm A, 2026-09-09)
    python3 - "$REC/soak-summary.json" <<'PY' || fail=1
import json, sys
d = json.load(open(sys.argv[1])); print("soak", {k: d.get(k) for k in ("ok","err","finish_reasons","health_after") if k in d})
sys.exit(0 if d.get("err", 1) == 0 and d.get("ok", 0) > 0 else 1)
PY
    local ph pw; ph=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^head=/){sub("head=","",$i); if($i+0>m)m=$i+0}} END{print m+0}' "$REC/soak-clocks.txt"); pw=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^worker=/){sub("worker=","",$i); if($i+0>m)m=$i+0}} END{print m+0}' "$REC/soak-clocks.txt")
    echo "soak_clock_peak_head=$ph" >>"$REC/facts.txt"; echo "soak_clock_peak_worker=$pw" >>"$REC/facts.txt"; log "gate 4: clock peaks under soak head=$ph worker=$pw (need >=2400)"; (( ph >= 2400 && pw >= 2400 )) || fail=1
    kill $mw 2>/dev/null; wssh 'kill $(cat /tmp/memwatch-gates.pid) 2>/dev/null; cat /tmp/memwatch-gates-worker.tsv' >"$REC/memwatch-gates-worker.tsv"
    local b1 b2 s1 s2; b1=$(nvrm_count "$g_since" "$g_ready"); b2=$(wssh "journalctl -k --since '$g_since' --until '$g_ready' 2>/dev/null | grep -c NV_ERR_NO_MEMORY"); s1=$(nvrm_count "$g_ready"); s2=$(wssh "journalctl -k --since '$g_ready' 2>/dev/null | grep -c NV_ERR_NO_MEMORY")
    { echo "nvrm_boot_head=$b1"; echo "nvrm_boot_worker=$b2"; echo "nvrm_postready_head=$s1"; echo "nvrm_postready_worker=$s2"; } >>"$REC/facts.txt"
    log "gate 5: NVRM boot $b1/$b2 post-ready $s1/$s2 (recorded)"
    python3 - "$REC/memwatch-gates-head.tsv" "$REC/memwatch-gates-worker.tsv" <<'PY' | tee -a "$REC/facts.txt"
import sys
for f in sys.argv[1:]:
    rows=[l.split("\t") for l in open(f).read().splitlines()[1:] if l]
    if rows: print(f"memavail_min_{'head' if 'head' in f else 'worker'}_gib={min(int(r[1]) for r in rows)/1048576:.2f}")
PY
    [[ -f "$RECORDS/LAUNCHES.tsv" ]] || echo -e "ts_utc\tprofile\tverdict\tkv_tokens\timage\trecord" >"$RECORDS/LAUNCHES.tsv"
    local kv; kv=$(grep -oE '^kv_pool_tokens=.*' "$REC/facts.txt" | cut -d= -f2)
    if (( fail )); then echo -e "$(TS)\t$PROFILE_NAME\tGATE_FAIL\t$kv\t$IMAGE_ID\t$REC" >>"$RECORDS/LAUNCHES.tsv"; log "gates: FAILED — stopping cluster"; do_stop; exit 5; fi
    echo -e "$(TS)\t$PROFILE_NAME\tPASS\t$kv\t$IMAGE_ID\t$REC" >>"$RECORDS/LAUNCHES.tsv"; log "gates: PASS"
}
do_stop() {
    [[ -n "$REC" && -d "$REC" ]] && docker logs "$CONTAINER" >"$REC/boot.log" 2>&1 || true
    [[ -d "$RECORDS/latest" ]] && docker logs "$CONTAINER" >"$RECORDS/latest/final.log" 2>&1 || true
    ( cd "$EUGR" && ./launch-cluster.sh stop ) >/dev/null 2>&1 || true
    pick_worker >/dev/null 2>&1 || true
    for i in 1 2 3 4 5 6; do cup || { wcup || break; }; sleep 5; done
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; [[ -n "$WORKER" ]] && wssh docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    log "stop: containers down (head=$(cup && echo up || echo down) worker=$(wcup && echo up || echo down))"
}
new_record() { REC="$RECORDS/$(date -u +%Y%m%dT%H%M%SZ)__$PROFILE_NAME"; mkdir -p "$REC"; ln -sfn "$REC" "$RECORDS/latest"; cp "$PROFILE_FILE" "$REC/profile.env"; log "record: $REC"; }
metrics() { curl -s -m 10 "http://localhost:${PORT:-8000}/metrics" 2>/dev/null | awk '
    /^vllm:num_requests_running/ {r+=$NF} /^vllm:num_requests_waiting/ {w+=$NF}
    /^vllm:prompt_tokens_total/ {p+=$NF} /^vllm:generation_tokens_total/ {g+=$NF}
    END {printf "%d %d %.0f %.0f\n", r, w, p, g}'; }
worker_check() { # 0 ok, 1 container gone (ssh ok), 2 unreachable
    local out; out=$(wssh docker ps -q --filter "name=^${CONTAINER}$" 2>/dev/null) || return 2
    [[ -n "$out" ]] || return 1; return 0
}
monitor() {
    # self-heal loop for systemd Type=exec: /health every 20 s (6 fails = DIED), worker
    # container every 100 s (gone = DIED; unreachable 3x), kernel NV_ERR every 300 s (warn
    # only — first-touch bursts are expected), and a wedge check: if requests are
    # running/waiting but neither prompt_tokens_total nor generation_tokens_total advanced
    # for WEDGE_CHECKS consecutive 300 s checks (5 = 25 min), the engine is hung with
    # /health still 200 — stop the pair and exit 2 (systemd restarts). 25 min exceeds a
    # full 1M-token prefill, so a legitimately busy engine is never killed.
    log "monitor: entering loop (health 20s; worker 100s; NVRM + progress 300s; wedge = ${WEDGE_CHECKS}x300s without token progress while busy)"
    local fails=0 tick=0 nv_last=0 wfails=0 stall=0 last_p=-1 last_g=-1
    while true; do
        sleep 20; (( tick++ )) || true
        if health_ok; then fails=0; else
            (( fails++ )) || true; log "monitor: /health failure $fails/6"
            cup || { log "monitor: head container gone"; do_stop; exit 2; }
            (( fails >= 6 )) && { log "monitor: /health dead 2min"; do_stop; exit 2; }
        fi
        if (( tick % 5 == 0 )); then
            local wrc=0; worker_check || wrc=$?
            case $wrc in
                0) wfails=0 ;;
                1) log "monitor: worker container gone"; do_stop; exit 2 ;;
                2) (( wfails++ )) || true; log "monitor: worker unreachable $wfails/3"
                   (( wfails >= 3 )) && { log "monitor: worker unreachable 5min"; do_stop; exit 2; } ;;
            esac
        fi
        if (( tick % 15 == 0 )); then
            local nv; nv=$(journalctl -k --since "@${READY_EPOCH:-$LAUNCH_EPOCH}" 2>/dev/null | grep -c 'NV_ERR_NO_MEMORY' || true)
            (( nv > nv_last )) && { log "monitor: WARNING kernel NV_ERR_NO_MEMORY count now $nv (was $nv_last)"; nv_last=$nv; }
            local m r w p g; m=$(metrics || echo "0 0 0 0"); read -r r w p g <<<"$m"
            if (( r + w > 0 )); then
                if [[ "$p" == "$last_p" && "$g" == "$last_g" ]]; then
                    (( stall++ )) || true; log "monitor: busy (running=$r waiting=$w) but no token progress — stall check $stall/$WEDGE_CHECKS"
                    (( stall >= WEDGE_CHECKS )) && { log "monitor: engine wedged ($((WEDGE_CHECKS*5)) min no progress while busy)"; do_stop; exit 2; }
                else stall=0; fi
            else stall=0; fi
            last_p=$p; last_g=$g
        fi
    done
}
case "${1:-}" in
    start)  PROFILE_FILE="${2:?profile.env}"; load_profile "$PROFILE_FILE"; new_record; preflight; render_serve; launch_stack; wait_ready; fill_record; gates; log "start: serving and gated — $REC" ;;
    serve)  PROFILE_FILE="${2:?profile.env}"; load_profile "$PROFILE_FILE"
            if health_ok && cup && pick_worker && wcup; then
                REC="$RECORDS/latest"; [[ -d "$REC" ]] || new_record
                READY_EPOCH=$(grep -oE '^ready_epoch=.*' "$REC/facts.txt" 2>/dev/null | cut -d= -f2); LAUNCH_EPOCH=$(grep -oE '^launch_epoch=.*' "$REC/facts.txt" 2>/dev/null | cut -d= -f2)
                : "${READY_EPOCH:=$(date +%s)}"; : "${LAUNCH_EPOCH:=$READY_EPOCH}"
                log "serve: adopting already-healthy pair (worker=$WORKER)"; monitor
            fi
            log "serve: no healthy pair — full relaunch"
            new_record; do_stop; preflight; render_serve; launch_stack; wait_ready; fill_record; gates
            log "serve: serving and gated — entering monitor"; monitor ;;
    gates)  PROFILE_FILE="${2:?profile.env}"; load_profile "$PROFILE_FILE"; pick_worker; REC="$RECORDS/latest"; [[ -d "$REC" ]] || new_record; READY_EPOCH=$(grep -oE '^ready_epoch=.*' "$REC/facts.txt" 2>/dev/null | cut -d= -f2); LAUNCH_EPOCH=$(grep -oE '^launch_epoch=.*' "$REC/facts.txt" 2>/dev/null | cut -d= -f2); : "${READY_EPOCH:=$(date +%s)}"; : "${LAUNCH_EPOCH:=$READY_EPOCH}"; health_ok || die 5 "server not healthy"; gates ;;
    stop)   PORT=8000; pick_worker || true; REC="$RECORDS/latest"; do_stop ;;
    status) PORT=8000; pick_worker || true; echo "head: $(docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep -E "$CONTAINER|dspark" || echo none)"; echo "worker: $(wssh docker ps --format "'{{.Names}} {{.Image}} {{.Status}}'" | grep -E "$CONTAINER|dspark" || echo none)"; echo "health: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:8000/health)"; [[ -f "$RECORDS/latest/facts.txt" ]] && grep -E 'profile|kv_pool|max_conc' "$RECORDS/latest/facts.txt" ;;
    *) echo "usage: $0 {start <profile.env>|serve <profile.env>|stop|gates <profile.env>|status}"; exit 64 ;;
esac
