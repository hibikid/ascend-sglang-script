#!/usr/bin/env bash
set -eo pipefail

# ticket03 GLM-5.2 mempool A/C demo：P eager，D Graph BS=16，原生 fake warmup 开启。
# SGLang 使用 f3e522f9b6 或包含该提交的后续版本。
# 先在 P 机另开终端运行 router-mempool-demo.sh，再启动两侧 worker：
#   31: LOCAL_HOST1=10.120.72.31 bash pd-disaggregation/glm52mempool-demo.sh
#   32: LOCAL_HOST1=10.120.72.32 bash pd-disaggregation/glm52mempool-demo.sh
# P 等 BM join 时即可启动 D，不要等 P ready。
# IP、模型、NUMA 和端口统一修改同目录 glm52mempool-demo.json，两机保持一致。
# CONFIG_FILE、RUN_ID、LOG_DIR 可覆盖；三个进程的 CONFIG_FILE 内容和 RUN_ID 必须相同。
# --dry-run 只打印启动参数，不加载模型或启动 NPU。

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
CONFIG_FILE=${CONFIG_FILE:-${script_dir}/glm52mempool-demo.json}
RUN_ID=${RUN_ID:-glm52-mempool-ac}
LOG_DIR=${LOG_DIR:-/home/cryang/log/mempool-demo/${RUN_ID}}
LOCAL_HOST1=${LOCAL_HOST1:-}
dry_run=${1:-}
if [[ $# -gt 1 || ( -n "$dry_run" && "$dry_run" != --dry-run ) ]]; then
    echo "Usage: LOCAL_HOST1=<P/D IP> bash $0 [--dry-run]" >&2
    exit 2
fi
CONFIG_FILE=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
cd "$SGLANG_DIR"
export PYTHONPATH="${PWD}/python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$dry_run" != --dry-run ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="/usr/local/Ascend/8.5.0/compiler/bishengir/bin:${PATH}"
fi

# 复用通过 NPU 验收的环境生成和 worker 启动脚本，保持容量/Graph/warmup 一致。
# 只启动本机 worker；Main Router 由 router-mempool-demo.sh 独立启动。
python3 - "$CONFIG_FILE" "$LOCAL_HOST1" "$RUN_ID" "$LOG_DIR" "$dry_run" <<'PY'
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

scripts = Path.cwd() / "ascend-mempool-test/scripts"
sys.path.insert(0, str(scripts))
from run_mempool_min import RouterConfig, service_environment

config_path, local_host, run_id, log_dir, dry_run = sys.argv[1:]
config = RouterConfig.read(config_path)
local_host = local_host or config.prefill_host
roles = {config.prefill_host: "prefill", config.decode_host: "decode"}
if local_host not in roles:
    raise SystemExit(f"LOCAL_HOST1={local_host} does not match the configured P/D hosts")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
    raise SystemExit("RUN_ID must be a simple name of at most 80 characters")
log_root = Path(log_dir).resolve()
if not log_root.is_relative_to(Path("/home/cryang/log").resolve()):
    raise SystemExit("LOG_DIR must remain under /home/cryang/log/")
role = roles[local_host]
args = SimpleNamespace(config=config_path, run_id=run_id, role=role)
env = service_environment(config, args, log_root, target=True)
command = ["bash", str(scripts / "run_service.sh"), role, "formal"]
if dry_run:
    command.append("--dry-run")
print(f"role={role} host={local_host} run_id={run_id} config={config_path}", flush=True)
print(f"prefix_pages={config.prefix_pages} match_prefix={config.match_prefix} verify_prefix={config.verify_prefix}", flush=True)
print(f"P=eager D=Graph(BS=16) warmup=enabled logs={log_root}", flush=True)
os.execvpe(command[0], command, env)
PY
