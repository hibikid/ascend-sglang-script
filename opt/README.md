# GLM-5.2 mempool 性能试跑

四个入口都用普通 `case` 解析参数。默认值在文件顶部；P/D 启动命令在
`glm52mempool-opt.sh` 下半部分，每条参数独占一行。运行 `--help` 查看接口。
Python 只处理配置、请求构造/结果保存、profile API，不隐藏模型启动参数。

依赖本轮 `cryang/dev/mempool-min-opt` 中的 SGLang 和 `ascend-mempool-test/scripts/opt_*.py`。
两机都要更新 SGLang 和本脚本仓库，使用现有 CANN/ATB/BM/UniDex 环境及 GLM-5.2 权重。
默认 SGLang 路径 `/home/cryang/sglang`，可用 `--sglang-dir` 或 `SGLANG_DIR` 修改。

## 1. 启动 Router、P、D

以下命令从 `ascend-sglang-script` 根目录执行，各进程各占一个终端。
默认 IP 是 P=10.120.72.31、D=10.120.72.32，网卡 bond4；模型路径见 JSON。
先核对实际机器和模型路径，再运行。P 开始等待 BM join 时即可启动 D。

```bash
# P 机，终端 1
bash opt/router-mempool-opt.sh --run-id opt-001

# P 机，终端 2
bash opt/glm52mempool-opt.sh --role prefill --run-id opt-001

# D 机，终端 1
bash opt/glm52mempool-opt.sh --role decode --run-id opt-001
```

三者使用相同 JSON、run-id 和覆盖参数。解析后的完整配置保存为
`/home/cryang/log/mempool-opt/opt-001/config.json`；P/D 启动前会与正在运行的
Router 核对配置。同一日志目录不能改容量；改容量时用新的 run-id。

例如自定义 IP、P alive 页数时，三条命令都追加同样的参数：

```bash
--p-ip 10.120.72.31 --d-ip 10.120.72.32 --palive-page 60
```

也可以编辑一份 `glm52mempool-opt.json`，复制到两机，然后三者传入 `--config FILE`。
端口、chunk、context、输出上限、HBM token 容量、NUMA 等低频选项集中在 JSON。
Router 和 worker 仍使用各自的前台日志：`router.log`、`prefill.log`、`decode.log`。

默认 page64、prefix192、P alive main=0、P Index=4096、D alive main=4096 页，
继承 shared pools/dual_write；observe、observe_trace、verify_prefix 全关。
P eager，D Graph BS=16，原生 warmup 开启；LRU=0、cache factor=2。
页数表示物理页预算，不是单请求 token 上限。dual_write 路径不消费 P alive main；
`--palive-page 60` 会增加其预算，但不会切换写入路径。
默认服务允许输入最多 8192、输出最多 512；context=8832、max-total-tokens=139520。
后续两张 baseline 固定 prefix192；临时试其他容量不作为那两张 baseline。

## 2. 产生负载

在有本地模型 tokenizer 的 P/D 任一机器另开终端：

```bash
# 单条冷输入，检查启动和请求链路
bash opt/requests.sh --input-len 4096 --output-len 128 --concurrency 1 --count 1

# 冷输入持续发出，共 64 条，最多 16 条同时请求
bash opt/requests.sh --input-len 4096 --output-len 128 --concurrency 16 --count 64

# 先完成一条种子请求，再测量 16 条同输入的 prefix reuse
bash opt/requests.sh --input-len 4096 --output-len 128 --mode reuse --count 16
```

使用其他 run-id 时增加 `--config /home/cryang/log/mempool-opt/<run-id>/config.json`。
输入长度包含 chat template，必须是页大小的倍数；固定 greedy，ignore_eos=true。
`--concurrency` 是客户端并发，是否达到实际活跃 batch 要看 worker 的 perf_step 日志。
`reuse` 的种子不计入 count/测量耗时。每条 cold 输入的首个 token 页不同。
若 prefix 容量装不下 reuse 输入，脚本会按实际 hit_len 报失败。

每轮保存 workload.json、config.json、每请求响应（含 Router TTFT/TPOT 与客户端耗时）、
before/after.json、report.json。失败返回非零；已完成的响应保存在对应目录。
这些计时口径沿用票30：TTFT 以 Router 收到首 token 为终点，客户端耗时是完整响应耗时。

```bash
# 重放同一份真实 token IDs；这些长度/数量/mode 必须与保存的负载匹配
bash opt/requests.sh --input-len 4096 --output-len 128 --concurrency 16 --count 64 \
    --workload /home/cryang/log/mempool-opt/requests/<上一轮时间>/workload.json
```

同服务重放 cold 可能已命中缓存，会报告失败；要比较冷负载，请在新服务实例上重放。

## 3. 打开采集窗口

先启动较长负载，看到 `Starting ... requests` 且 worker 正在 decode 后，在 **D 本机**运行：

```bash
bash opt/profiler.sh --ranks 0,1 --seconds 2 \
    --output-dir /home/cryang/log/mempool-opt/traces/d-rank01-001

# P 本机采集时改为 P worker URL
bash opt/profiler.sh --url http://10.120.72.31:8000 --ranks 0 --seconds 2 \
    --output-dir /home/cryang/log/mempool-opt/traces/p-rank0-001
```

`--ranks all` 采全部 16 个 TP rank；默认仅 rank0。这里是该单机服务的 TP rank，
不是 OS PID 或 NUMA 编号。本轮启动方式 TP16/base-gpu-id0/gpu-id-step1，rank 与设备号对应。
真正选择采集 rank；未选 rank 继续推理，只在停止时参加原有同步。
仅支持 NPU V1 手动窗口，不与 by-stage、num-steps 或其他 profile 客户端同时使用。

seconds 是 start API 返回后的等待时长；启停有开销，实际设备窗口不保证精确到该秒数。
stop 的导出耗时单独记录在 window.json。Ctrl-C 会尝试 stop 并保存窗口记录。
输出目录必须是新目录，且在服务本机可见；该脚本不提供下载功能。
采到空的设备窗口仍可能导出 trace 文件，因此需在离线查看中确认负载/设备事件确实落入窗口。

从输出下的 `*/ASCEND_PROFILER_OUTPUT/trace_view.json` 取 trace，可尝试在 Mac 的 Perfetto UI 打开。
若字符串时间戳/超大 flow ID 不兼容，按 TICKET30 的 viewer 说明使用 MindStudio Insight；
完整 NPU trace 目录仍应保留。严格 W-0 的设备事件与容量检查仍使用
SGLang 仓库的 `ascend-mempool-test/TICKET30.md` 入口。

## 4. 本地预览与验证边界

```bash
bash opt/glm52mempool-opt.sh --role decode --palive-page 60 --dry-run
bash opt/router-mempool-opt.sh --palive-page 60 --dry-run
bash opt/requests.sh --config opt/glm52mempool-opt.json --concurrency 16 --count 16 --dry-run
bash opt/profiler.sh --ranks 0,1 --seconds 2 --dry-run
```

Mac 预览时把 `--sglang-dir` 指向本地 SGLang 工作树，并使用装有 msgspec/torch 等项目依赖的 Python。
这些入口用于交互式试跑/profile；不自动关停用户启动的服务，也不替代票30 W-0 完整验收。
硬件运行由用户执行。首次回传对应 run-id 的 config、三侧日志、请求结果目录与完整 trace 目录。

## trace-001 导出故障后的更新

opt启动设置SGLANG_NPU_MEMPOOL_PROFILE_RAW=1，服务内只采集落盘，避免自动解析fork触发BM清理崩溃。
profiler.sh在stop返回后由独立客户端解析，默认max_process_number=1；window.json记录开停，parse.json记录解析。
该改动要在更新两个仓库并重新启动服务后生效；已有trace不必重采。失败的已有原始目录可用：

```bash
bash opt/profiler.sh --parse-only --output-dir /home/cryang/log/mempool-opt/trace-001/trace-d0
```

用户已验证独立解析恢复176.9MiB trace，总解析46.60秒。新服务采集流程仍待NPU确认。
