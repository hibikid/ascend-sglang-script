# 旧 naive mempool：独立 opt 入口

这套脚本对应旧版 `cryang/dev/mempool` / `d328af12bd`：每个TP rank独立池、固定16 slots、prompt KV写P DRAM、生成KV写D DRAM，miss分两次UniDexCopy读取。
当前性能分支只提供工具；实际模型从旧版独立checkout加载，并复用票31的rank选择/raw-only采集补丁。

沿用上层opt的脚本约定：默认值在顶部，普通case解析，`--help`和`--dry-run`，模型启动命令在shell中逐项展开。
Python只处理配置、checkout、请求和报告；P/D完整stdout/stderr先tee到日志，再按开关过滤终端。
本路径使用旧版原生PD MiniLB Router，负责选择P/D并派发请求。启动入口为router-mempool-opt.sh。

## 默认值

- 工具仓库：`/home/cryang/sglang`，分支`cryang/dev/mempool-min-opt`；`--sglang-dir`可覆盖。
- 旧版checkout：`/home/cryang/sglang-naive-profile`；`--naive-sglang-dir`可覆盖。
- 配置：本目录`glm52mempool-opt.json`；P/D为31/32，模型GLM-5.2-w4a8，TP16。
- 日志：`/home/cryang/log/mempool-opt/naive-base/<run-id>/`，默认run-id `naive-001`。
- page64，P eager，D Graph BS16；每P slot预留8192 tokens，每D slot预留8704 tokens，单请求输出上限512。
- `decode_capacity`是D slot的实际分配长度；默认8704与票31已有旧版launch相同，不是请求输出长度。旧版没有当前的prefix/物理页预算配置。

只运行当前或旧版其中一组服务。切换前由用户按原方式结束已有P/D服务，释放NPU和端口。
两台更新SGLang及本脚本仓库；使用现有CANN/ATB/BM/UniDex环境。服务器操作由用户执行。

## 1. 两台分别准备一次

从脚本仓库根目录执行：

```bash
bash opt/naive-base/prepare.sh --run-id naive-001
```

创建/复用旧版checkout，保存本机`config.json`及`naive-checkout.json`，不加载模型。
已有checkout须是上述固定版本并带票31补丁；不覆盖其他版本或删除已有目录。
P/D须使用相同JSON和覆盖参数。可编辑本目录JSON复制到两机，或在两边prepare时传相同参数：

```bash
bash opt/naive-base/prepare.sh --run-id naive-002 \
  --p-ip 10.120.72.31 --d-ip 10.120.72.32 \
  --prompt-capacity 8192 --decode-capacity 8704
```

prepare参数：IP、模型、网卡、page-size、P/D slot容量，以及config、run-id、log-dir、两个仓库路径。
不需要先启动当前版服务，也不需要已有ticket31-current目录。
prepare是一次性命令，打印`[prepare] DONE`后自动退出，原终端所在目录与当前仓库分支保持不变；无需手动切回来。

## 2. 启动Router、P/D

P机Router终端：

```bash
bash opt/naive-base/router-mempool-opt.sh --run-id naive-001
```

默认绑定P_IP:6699；改端口用`--port`，请求端对应传`--router-url`。
Router和P/D都常驻前台，各占一个终端。

P机服务终端：

```bash
bash opt/naive-base/glm52mempool-opt.sh --role prefill --run-id naive-001
```

P等待BM join时，在D终端启动：

```bash
bash opt/naive-base/glm52mempool-opt.sh --role decode --run-id naive-001
```

默认读取prepare保存的配置。启动同一个run时不可变更配置；改配置请用新run-id并再次prepare。
可增加`--mempool-console 1`显示全部细节；默认0保留原生性能输出和错误。
`SGLANG_MEMPOOL_CONSOLE`可设置默认值，CLI优先；两端独立选择。
保存`router.log`、`prefill.log`、`decode.log`、各端launch命令及identity文件（旧SHA、补丁diff、依赖/模型元数据等）。
启动时尚无workload，identity中的workload hash为null；每轮请求保存自己的workload hash。

## 3. D本机发请求

```bash
bash opt/naive-base/requests.sh \
  --input-len 4096 --output-len 128 --concurrency 1 --count 5
```

其他run增加`--config /home/cryang/log/mempool-opt/naive-base/<run-id>/config.json`。
可覆盖`--output-dir`、`--timeout`、`--decode-log`、`--router-url`和`--sglang-dir`。
请求默认发到P_IP:6699的Router，Router分配bootstrap room并转发P/D。
默认读取config同目录的`decode.log`，完成后核对本轮新增room数量与请求数相符，且每个room有16rank的RELEASE_ACK/CLOSED。
原生Router不回传room与客户端rid映射，所以这里是整轮释放核对；请在专用服务上运行，不混入其他请求。
每条请求固定greedy/ignore_eos，输入含模板且为页大小倍数；保存完整响应、输入IDs、TTFT/TPOT和报告。
失败返回非零，已完成的响应保留；客户端并发数不代表实际执行批宽。

复用当前`opt/requests.sh`保存的同一组IDs，可用于同负载对照：

```bash
bash opt/naive-base/requests.sh \
  --input-len 4096 --output-len 128 --concurrency 1 --count 5 \
  --workload /home/cryang/log/mempool-opt/requests/<当前版那一轮>/workload.json
```

workload长度、输出和case数量必须匹配命令；不会再次tokenize。
旧版没有跨请求prefix复用，因此不提供`--mode reuse`。当前opt的cold负载可原样重放；复用模式的单case文件不能冒充多case负载。
旧版TTFT从客户端经Router派发到收到D首token，当前版Router TTFT起于reserve；E2E边界也不同，不直接相减。
首chunk多token时exact TTFT为空，另记first chunk。人工开trace期间的请求成绩不作为无trace性能基线。

## 4. D本机采集

先运行足够长的请求，看到decode在推进后，另开D终端：

```bash
bash opt/naive-base/profiler.sh --ranks 0 --seconds 2 \
  --output-dir /home/cryang/log/mempool-opt/naive-base/trace-001
```

接口与上层opt相同：`--url`、`--ranks 0,1|all`、`--seconds`、`--output-dir`、`--timeout`、`--sglang-dir`。
P端采集需在P本机改worker URL。服务只落raw数据，独立客户端解析；失败可复用原目录：

```bash
bash opt/naive-base/profiler.sh --parse-only \
  --output-dir /home/cryang/log/mempool-opt/naive-base/trace-001
```

参考trace-002为2秒/rank0约699MiB；旧版事件密度可能不同，实际大小以采集结果为准。
保留整个profiler目录；stop和parse时间分别记录。文件导出成功不等于窗口内有完整模型step。

## 5. 本地预览与范围

```bash
bash opt/naive-base/prepare.sh --dry-run
bash opt/naive-base/router-mempool-opt.sh --dry-run
bash opt/naive-base/glm52mempool-opt.sh --role decode --mempool-console 1 --dry-run
bash opt/naive-base/requests.sh --config opt/naive-base/glm52mempool-opt.json --count 5 --dry-run
bash opt/naive-base/profiler.sh --ranks 0 --seconds 2 --dry-run
```

Mac预览增加`--sglang-dir <当前工具仓库>`，使用装有项目依赖的Python。
所有入口均不在dry-run中启动服务或新建checkout。
本套用于交互式性能试跑；票31固定warmup/S1/W0、当前S1/S2/S3及正式汇总继续使用SGLang的`ascend-mempool-test/TICKET31.md`。
新增配置JSON的decode_capacity语义是旧版固定slot容量，不要将其直接传给当前Router的配置入口。
