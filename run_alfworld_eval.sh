#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--dataset alfworld|pddl] [comem|g-memory]\n' "$0"
}

dataset=alfworld
method=
while (($#)); do
    case "$1" in
        --dataset)
            if (($# < 2)); then
                usage >&2
                exit 2
            fi
            dataset=$2
            shift
            ;;
        comem|g-memory)
            if [[ -n $method ]]; then
                usage >&2
                exit 2
            fi
            method=$1
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done
case "$dataset" in
    alfworld) qwen_port=18452 ;;
    pddl) qwen_port=18453 ;;
    *) usage >&2; exit 2 ;;
esac
methods=(comem g-memory)
if [[ -n $method ]]; then
    methods=("$method")
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /home/xuyining/miniconda3/etc/profile.d/conda.sh
conda activate gmemory

qwen_launcher=/data1/xuyining/conda-envs/GMemory-assets/qwen-local/start-qwen.sh
model=Qwen/Qwen2.5-14B-Instruct
endpoint="http://127.0.0.1:$qwen_port"
state_root="$PWD/.db/$dataset-autogen-eval"
log_dir="$PWD/my_logs/$dataset/autogen/qwen2.5-14b-instruct"

export OPENAI_API_BASE="$endpoint/v1" OPENAI_API_KEY=local-gmemory GMEMORY_LLAMA_CPP=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY" PYTHONUNBUFFERED=1

mkdir -p -- "$state_root"
exec 9>"$state_root/eval.lock"
if ! flock -n 9; then
    printf 'An AutoGen evaluation for %s is already running.\n' "$dataset" >&2
    exit 1
fi
if [[ -n $(ss -H -ltn "( sport = :$qwen_port )") ]]; then
    printf 'Port %s is already in use. Stop that service before starting this evaluation.\n' "$qwen_port" >&2
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

printf 'Local Qwen: %s\n' "$endpoint"
GMEMORY_QWEN_PORT="$qwen_port" "$qwen_launcher" > "$state_dir/qwen-server.log" 2>&1 9>&- &
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
    setsid python -u tasks/run.py --task "$dataset" --mas_type autogen \
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
