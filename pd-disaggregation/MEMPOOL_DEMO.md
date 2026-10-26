# GLM-5.2 mempool-min A/C 语义检查

需要 SGLang `f3e522f9b6` 或包含它的后续版本。两机使用同一版本及同一份
`glm52mempool-demo.json`。默认路径为 `/home/cryang/sglang`。

本次只检查库存题的冷算 A 与复用 C。配置启用 `prefix_pages=40`、
`match_prefix=true`，关闭 `verify_prefix` 和临时算子诊断。
正常恢复不注入验收用的 allocator 空洞及延迟 fill。
P eager、D Graph BS=16、page_size=128、SP=8192、SD=128。

按用户要求保留三个独立前台服务终端。先结束上一轮 demo 的这三个进程，
使用新的 RUN_ID。不要同时向该服务提交其他请求。

在两机的 `/home/cryang/ascend-sglang-script` 中运行。

31，终端一：

```bash
RUN_ID=ticket03-ac-001 bash pd-disaggregation/router-mempool-demo.sh
```

31，终端二：

```bash
RUN_ID=ticket03-ac-001 LOCAL_HOST1=10.120.72.31 bash pd-disaggregation/glm52mempool-demo.sh
```

32，终端一：

```bash
RUN_ID=ticket03-ac-001 LOCAL_HOST1=10.120.72.32 bash pd-disaggregation/glm52mempool-demo.sh
```

P 等待 join 时即可启动 D，无需等待 P ready。
三处 RUN_ID 必须相同；CONFIG_FILE、SGLANG_DIR、LOG_DIR 支持环境变量覆盖。
两份启动脚本支持 `--dry-run`，只打印配置与命令。

31，终端三：

```bash
RUN_ID=ticket03-ac-001 python3 pd-disaggregation/request-mempool-demo.py
```

客户端自动等待 P、D、Main 就绪，确认没有历史请求或 prefix，然后只发送 A/C。
它用本地 tokenizer 的 chat template，关闭 thinking，将空白 token 插在 user
内容开头，保留完整问题和 generation prompt，固定为 4096 input IDs。
最多生成 128 token，允许自然 EOS。A/C 使用完全相同的请求。
它核对 Router 的请求身份、释放状态和实际 hit_len=0/3968。

库存题内容见 `mempool-demo-prompt.txt`，以下答案只存在客户端中，不会发给模型：

```text
before=110,60,37
accept=O1,O3,O4
reject=O2,O5
after=35,10,2
buy=45,40,28
reason=O2:B;O5:A
```

终端打印两次完整回答，以及 `correct` / `incorrect` / `unparsed`。
`unparsed` 表示格式无法自动核对，需要人工读原文，不能直接归因为模型算错。
截断、错误答案、格式无法核对或命中/释放异常均非零退出。
只有两次完整答案都正确且命中检查通过，才报告语义 smoke `passed`。
output IDs 是否相同另行报告，不作为本语义检查的通过条件。
原 ticket03 的严格全输出验收标准没有改变。

结果保存在 `/home/cryang/log/mempool-demo/<RUN_ID>/ac/`：
`request.json`、完整编码后的 `rendered-prompt.txt`、`response-A/C.json`、
`response-A/C.txt`、`status-before/A/C.json`、`expected.txt` 和 `report.json`。
日志目录已存在时拒绝覆盖。先读取报告，再手动结束三个前台服务。
客户端不拥有服务进程，因此不关闭用户手动启动的服务。

`--prepare-only` 只编码请求，写到独立的 `ac-prepared/`，不发送 HTTP 请求。
再次测冷算必须重启三个服务并更换 RUN_ID，仅更换客户端 RUN_ID 不会清空缓存。

这是单题语义 smoke，不是模型数据集精度、性能或 ticket03 完整验收。
这也不是原 A/B/C 合成 prompt 的复现：省略 B、采用库存题及正常恢复路径。
