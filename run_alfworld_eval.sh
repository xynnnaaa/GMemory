#!/usr/bin/env bash
set -euo pipefail

methods=(comem g-memory)
case "$#:${1:-}" in
    0:) ;;
    1:comem|1:g-memory) methods=("$1") ;;
    *) printf 'Usage: %s [comem|g-memory]\n' "$0" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /home/xuyining/miniconda3/etc/profile.d/conda.sh
conda activate gmemory

qwen_launcher=/data1/xuyining/conda-envs/GMemory-assets/qwen-local/start-qwen.sh
model=Qwen/Qwen2.5-14B-Instruct
endpoint=http://127.0.0.1:18452
state_root="$PWD/.db/alfworld-autogen-eval"
log_dir="$PWD/my_logs/alfworld/autogen/qwen2.5-14b-instruct"

export OPENAI_API_BASE="$endpoint/v1" OPENAI_API_KEY=local-gmemory
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY" PYTHONUNBUFFERED=1

mkdir -p -- "$state_root"
exec 9>"$state_root/eval.lock"
if ! flock -n 9; then
    printf 'An AutoGen + ALFWorld evaluation is already running.\n' >&2
    exit 1
fi
if [[ -n $(ss -H -ltn '( sport = :18452 )') ]]; then
    printf 'Port 18452 is already in use. Stop that service before starting this evaluation.\n' >&2
    exit 1
fi

state_dir=$(mktemp -d "$state_root/$(date +%Y%m%d-%H%M%S).XXXXXX")
mkdir -p -- "$log_dir"
printf 'Evaluation logs: %s\n' "$log_dir"

server_pid=
eval_pid=
cleanup() {
    if [[ -n $eval_pid ]]; then
        kill -TERM -- "-$eval_pid" 2>/dev/null || kill -TERM "$eval_pid" 2>/dev/null || true
        wait "$eval_pid" 2>/dev/null || true
    fi
    if [[ -n $server_pid ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$qwen_launcher" > "$state_dir/qwen-server.log" 2>&1 9>&- &
server_pid=$!
ready=false
for ((attempt = 0; attempt < 60; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        break
    fi
    if curl --noproxy 127.0.0.1 --fail --silent --max-time 2 "$endpoint/health" >/dev/null; then
        ready=true
        break
    fi
    sleep 2
done
if [[ $ready != true ]]; then
    printf 'Qwen did not become ready. See %s/qwen-server.log\n' "$state_dir" >&2
    exit 1
fi

status=0
for method in "${methods[@]}"; do
    log_file="$log_dir/$method.log"
    printf 'Starting %s. Log: %s\n' "$method" "$log_file"
    setsid python -u tasks/run.py --task alfworld --mas_type autogen \
        --mas_memory "$method" --reasoning io --max_trials 30 --model "$model" \
        --working_dir "$state_dir/$method" --log_to_stderr \
        > "$log_file" 2>&1 9>&- &
    eval_pid=$!
    if wait "$eval_pid"; then
        printf 'Finished %s.\n' "$method"
    else
        code=$?
        printf '\nEvaluation exited with status %s.\n' "$code" >> "$log_file"
        printf '%s failed (exit %s). See %s\n' "$method" "$code" "$log_file" >&2
        status=1
    fi
    eval_pid=
done
exit "$status"
