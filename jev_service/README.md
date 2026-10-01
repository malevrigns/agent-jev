# AgentJev：可直接调用的决策模型

工作台：<http://127.0.0.1:8147/>（本地 SSH 转发运行时）。服务器独立服务监听 `127.0.0.1:18765`，运行在 GPU 1；不调用 27B，不修改原训练日志。

## 本版已经实现

- 一次提交多个状态和问题，直接返回完整概率分布，生成 token 为 0。
- Boolean：TRUE/FALSE 判断；Choice：2–255 个动态候选；Score：2–10 个有序等级及期望分数。
- 完整候选文本进入模型，不再只识别 read/edit 之类的工具类别。候选也可以是包含动作参数的 JSON 对象。
- 问题与选项 ID 只用于对应响应，不进入模型；实际含义必须放在问题和候选描述里。
- 输入路径上限取决于加载的 checkpoint：不带 `config` 键的 checkpoint（例如 quickstart 的 v1 发版权重）最多 2,048 tokens，超长时明确拒绝，不静默截断任务或候选；带 `config` 键的训练产物 checkpoint 走训练管线 collate，按 checkpoint 内的 `max_len`（常见为 512）静默截断过长的 state/question/candidate，并在响应的 `usage.truncated_inputs` 中报告计数。同一 checkpoint 升级前后的评分可能因此不同，升级训练权重时请核对该配置。训练模式下 `--encoder` 等 CLI 参数由 checkpoint 内配置覆盖。
- 独立候选路径微批处理；长前缀、多候选时自动复用每个问题的 KV 前缀，候选分支互不读取，最后统一通过已训练的集合评分头。
- 网页输入、概率条、原始 API 请求/响应；独立服务可被其他程序调用。

## 使用

```python
import requests

result = requests.post('http://127.0.0.1:8147/api/evaluate', json={
    'state': 'Current patch test results: 23 passed, 2 failed.',
    'questions': [
        {'id': 'done', 'type': 'boolean',
         'question': 'Has the task been completed (full test suite passing)?'},
        {'id': 'next', 'type': 'choice',
         'question': 'Which action is most useful given the latest test result?',
         'options': {
             'inspect': 'Read the two failing assertions and their implementation.',
             'submit': 'Submit the patch without fixing the failing tests.'
         }}
    ]
}, timeout=60).json()
print(result['results'][0]['answers'])
```

多个状态放在 `{"requests": [{"state": ..., "questions": ...}, ...]}` 中。Score 使用 `levels` 数组，返回 `score = Σ index × probability`，索引从 0 开始。Boolean 可用 `criteria: {"true": "...", "false": "..."}` 自定义真假含义。对象状态使用稳定 JSON 序列化。

这里输出的是已训练模型的判断分布。Choice 是给定候选集内的相对偏好，不是各动作独立成功率；需要绝对成功概率时，应逐动作分别构造 Boolean 问题，并在目标领域校准。服务本身不执行候选、不宣称任意领域已校准。

## 权重与验证

当前加载现有 `runs/phase4/final.pt`，不是随机头或让 27B 假扮 Jev。Phase 5 的已有验收里，策略价值没有提升且 stall 判断坍缩，因此这版采用较稳定的 Phase 4 通用权重。新的专用 routing 权重不适用于这个通用服务，会被拒绝加载。

本轮是服务与推理架构实现，没有把它描述成又一次模型训练。已有权重尚未直接对比 NanoJev 公布的同版本权重，因此没有声称决策质量超过 NanoJev。

- `tests/test_contract.py`：11 项契约测试通过。
- `verify_http.py` / `verification.json`：真实 GPU HTTP 检查；三种问题、255 候选、重排、问题隔离、超长拒绝。
- `verify_prefix.py` / `prefix_verification.json`：与原训练推理路径序列化及输出核对，短输入输出差异为 0。
- 固定长状态负载（64 个 Choice 候选 + 1 个 Boolean 问题）：原路径中位 609.65 ms，共享前缀自动模式 298.91 ms，约 2.04 倍。骨干输入 token 从 33,547 降到 2,551；概率最大绝对差异 0.000508，最终选项不变。三次计时、预热后测量；仅代表该合成负载，不是普遍速度保证或决策能力提升。
- HTTP 问题隔离 BF16 数值差异约 0.00282，属于本次测量结果，不声称逐位一致。

当前服务共享的是同一问题的候选前缀；不同问题之间尚未共享状态缓存。训练代码的 `TreeEncoder` 已实现可反向传播的前缀树编码，通过 `encoder_impl: tree` 启用；它会在分段因果 SDPA、稠密祖先掩码和独立路径之间选择，并提前在 CPU 构建树。上面的服务端 KV 复用测速不代表训练树编码器的速度，独立结果见 [TreeEncoder benchmark 报告](../outputs/tree_benchmark_report.md)。

## 启动与恢复

服务器项目根目录：

```bash
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python3 -m jev_service.server --port 18765
```

本地转发（已有转发时不要重复启动）：

```powershell
ssh -N -p 221 -o BatchMode=yes -o ExitOnForwardFailure=yes -L 8147:127.0.0.1:18765 root@172.30.60.216
```

`GET /health` 返回模型名称、权重 SHA256、接口限制和当前编码方式。日志：服务器 `jev_service/service.log`。服务当前是独立后台进程，重启服务器后按上面命令恢复。

## 接下来的模型主线

目标保持为自己的 Jev，不以负面论文或归档作为交付。下一轮应直接训练“状态 + 完整候选”的决策，而不是继续调整固定工具名分类阈值：

1. 固定 NanoJev 代码和权重版本，使用相同输入、候选和环境做直接比较；现有反复查看的 13 题只能作为开发集。
2. 对可执行的完整候选分支收集真实结果、时间和成本。分别保留相对选择目标与每个候选的独立成功计数，不能把后续大模型修好的结果无差别归功于第一步动作。
3. 按任务/场景划分训练、开发、校准和冻结测试集。先根据开发集迭代决策质量，再开启一次冻结测试；保留基线而不更换评价口径。
4. 使用本服务作为统一入口，替换权重后即可继续比较，不必重新实现运行时或把工具类别映射成猜测的动作参数。

参考核对：[NanoJev 项目](https://github.com/TianyuCodings/NanoJev)、[输入契约](https://github.com/TianyuCodings/NanoJev/blob/main/docs/TYPESAFE_CONTRACT.md)。本版是 AgentJev 的独立接口，没有宣称与 TypeSafe/NanoJev API 完全兼容。
