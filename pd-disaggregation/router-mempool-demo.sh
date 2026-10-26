#!/usr/bin/env bash
set -eo pipefail

# ticket03 mempool A/C demo Main Router，在 P 机的独立终端启动。
# 使用 SGLang 内的 Main/SR 协议；不要同时启动普通 router.sh 占用 6699。
# 启动顺序：本脚本 -> P worker -> D worker；三个进程保持前台运行。
# 共用 glm52mempool-demo.json 与 RUN_ID，默认服务为 http://10.120.72.31:6699。
# 发送请求前确认 P:8000/ready、D:8001/ready 和 Main:6699/ready 均返回 200。
# 接口为 /generate：greedy、stream=false、128 对齐的 input_ids、生成上限 128。
# 启动后运行 python3 pd-disaggregation/request-mempool-demo.py，自动发送 A/C。
# 只检查库存题语义；完整验收仍使用 run_mempool_min.sh --mode reuse-acceptance。

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
CONFIG_FILE=${CONFIG_FILE:-${script_dir}/glm52mempool-demo.json}
RUN_ID=${RUN_ID:-glm52-mempool-ac}
LOG_DIR=${LOG_DIR:-/home/cryang/log/mempool-demo/${RUN_ID}}
dry_run=${1:-}
if [[ $# -gt 1 || ( -n "$dry_run" && "$dry_run" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
CONFIG_FILE=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
cd "$SGLANG_DIR"
export PYTHONPATH="${PWD}/python${PYTHONPATH:+:${PYTHONPATH}}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [[ "$dry_run" != --dry-run ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="/usr/local/Ascend/8.5.0/compiler/bishengir/bin:${PATH}"
fi
python3 - "$CONFIG_FILE" "$RUN_ID" "$LOG_DIR" <<'PY'
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "ascend-mempool-test/scripts"))
from run_mempool_min import RouterConfig

config = RouterConfig.read(sys.argv[1])
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", sys.argv[2]):
    raise SystemExit("RUN_ID must be a simple name of at most 80 characters")
if not Path(sys.argv[3]).resolve().is_relative_to(Path("/home/cryang/log").resolve()):
    raise SystemExit("LOG_DIR must remain under /home/cryang/log/")
print(f"prefix_pages={config.prefix_pages} match_prefix={config.match_prefix} verify_prefix={config.verify_prefix}")
print(f"Main: http://{config.prefill_host}:{config.main_port}  SR: {config.control_endpoint}")
PY

command=(python3 -u -m sglang.srt.disaggregation.ascend.mempool_router
    --config "$CONFIG_FILE" --run-id "$RUN_ID")
if [[ "$dry_run" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$LOG_DIR"
"${command[@]}" 2>&1 | tee "$LOG_DIR/router.log"
